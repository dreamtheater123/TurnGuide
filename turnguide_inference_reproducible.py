"""
This file is the GLM-4-Voice inference code for generating double channel audio.
"""

import os
import sys
import argparse

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(REPO_ROOT)
sys.path.append(os.path.join(REPO_ROOT, "third_party", "Matcha-TTS"))

cuda_num = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_num)

DEFAULT_MODEL_PATH = "qqjz/turnguide_loss_2_1"
DEFAULT_TOKENIZER_PATH = "THUDM/glm-4-voice-tokenizer"
DEFAULT_FLOW_PATH = os.path.join(REPO_ROOT, "glm-4-voice-decoder")

temperature = 1.3  # try 0.8, 0.9, 1.0, and 1.3
p_value = 0.9
chunk_size = 5

import torchaudio
import torch
import json
from datetime import datetime
from tqdm import tqdm
import uuid
from transformers import WhisperFeatureExtractor, AutoTokenizer, AutoModel, BitsAndBytesConfig
from speech_tokenizer.modeling_whisper import WhisperVQEncoder
from speech_tokenizer.utils import extract_speech_token
from flow_inference import AudioDecoder


special_token_ids = {
    "<|begin_of_audio|>": 151343, 
    "<|end_of_audio|>": 151344, 
    "<|system|>": 151335, 
    "<|user|>": 151336, 
    "<|assistant|>": 151337,
    "<|audio_0|>": 152353,
    "<|begin_of_transcription|>": 151345, 
    "<|end_of_transcription|>": 151346,
    "<|reserved_151347|>": 151347,  # serves as speaker1 id token
    "<|reserved_151348|>": 151348,  # serves as speaker2 id token
    "<|reserved_151349|>": 151349,  # serves as the end of text token
}


def check_speech_token_num(token_sequence, audio_offset):
    # First, find the first occurrence of consecutive tokens >= audio_offset
    i = 0
    while i < len(token_sequence):
        if token_sequence[i] >= audio_offset:
            # Found start of consecutive tokens >= audio_offset
            start_index = i
            count = 0
            
            # Count consecutive tokens >= audio_offset
            while i < len(token_sequence) and token_sequence[i] >= audio_offset:
                count += 1
                i += 1
            
            # We found the first group of consecutive tokens >= audio_offset
            if count >= chunk_size:
                # Trim to include exactly chunk_size tokens
                token_sequence = token_sequence[:start_index + chunk_size]
                return True, token_sequence
            else:
                # Trim to include all consecutive tokens found
                token_sequence = token_sequence[:start_index + count]
                return False, token_sequence
        else:
            i += 1
    
    # No consecutive tokens >= audio_offset found
    return False, token_sequence


class SpeechProcessor:
    def __init__(
        self,
        model_path="./glm-4-voice-9b",
        tokenizer_path="./glm-4-voice-tokenizer",
        flow_path="./glm-4-voice-decoder",
        device="cuda",
        dtype="bfloat16"
    ):
        self.device = torch.device(device)
        
        # Initialize GLM model
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        ) if dtype == "int4" else None

        print("Loading GLM model...")
        self.glm_model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if dtype == "bfloat16" else None,
            device_map="auto"
        ).eval()
        self.glm_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # Initialize speech tokenizer
        print("Loading Whisper model (Speech Tokenizer)...")
        self.whisper_model = WhisperVQEncoder.from_pretrained(tokenizer_path).eval().to(device)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(tokenizer_path)

        # Initialize audio decoder
        print("Loading audio decoder...")
        flow_config = os.path.join(flow_path, "config.yaml")
        flow_checkpoint = os.path.join(flow_path, 'flow.pt')
        hift_checkpoint = os.path.join(flow_path, 'hift.pt')
        self.audio_decoder = AudioDecoder(
            config_path=flow_config,
            flow_ckpt_path=flow_checkpoint,
            hift_ckpt_path=hift_checkpoint,
            device=device
        )
        # print("sleeping forever...")
        # time.sleep(2**31 - 1)  # Sleep for ~68 years (max 32-bit signed int)
    
    def generate_conditional_interleaved_speech_token_only(
        self,
        input_audio_path,
        temperature,
        top_p,
        output_dir,
        assistant_audio_name="assistant.wav",
        stereo_audio_name="stereo_user_left_assistant_right.wav",
    ):
        """
        Process speech in an interleaved fashion to create full duplex dialogue:
        1. Extract the user's input audio into tokens
        2. For each user chunk, append it to sequence and generate one model chunk
        3. Continue until all user chunks are used
        4. Decode model chunks to speech
        """
        mono_output_path = os.path.join(output_dir, assistant_audio_name)
        stereo_output_path = os.path.join(output_dir, stereo_audio_name)
        if os.path.exists(mono_output_path) and os.path.exists(stereo_output_path):
            print(f"Mono and stereo output files {mono_output_path} and {stereo_output_path} already exist. Skipping generation.")
            return None, None, "exist"
        
        # Load the audio and extract speech tokens
        # We probably don't need to load the audio ourselves
        user_audio_tokens = extract_speech_token(
            self.whisper_model,
            self.feature_extractor,
            [input_audio_path]
        )[0]

        # NOTE: check if there are any unsupported tokens (debug here)
        if 16383 in user_audio_tokens:
            print(f"16383 in user_audio_tokens. Skipping...")
            return None, None, "unsupported_token"

        user_channel_chunks = [user_audio_tokens[i:i+chunk_size] for i in range(0, len(user_audio_tokens), chunk_size)]
        if len(user_channel_chunks[-1]) < chunk_size:
            print(f"Last user chunk with size less than chunk_size ({chunk_size}), removing...")
            user_channel_chunks = user_channel_chunks[:-1]
        print(f"User audio channel chunks extracted: {len(user_channel_chunks)} chunks")

        audio_offset = self.glm_tokenizer.convert_tokens_to_ids('<|audio_0|>')
        end_of_audio_offset = self.glm_tokenizer.convert_tokens_to_ids('<|end_of_audio|>')
        
        # Prepare initial prompt
        prompt = f"<|begin_of_audio|><|begin_of_audio|>"
        # Tokenize initial prompt
        inputs = self.glm_tokenizer([prompt], add_special_tokens=False, return_tensors="pt").to(self.device)
        input_ids = inputs['input_ids']
        
        # Track generated model tokens separately
        # user_audio_tokens = []
        output_model_audio_tokens = []
        user_texts, model_texts = "", ""
        overall_text = {"interleaved_text": []}
        
        # Process tokens in an interleaved fashion (this is chunk-level interleaving)
        # for conditional chunk interleaving, <|user|> 5 speech tokens <|assistant|> is given, only let the model to generate the model's audio tokens
        with torch.no_grad():
            global_flag = False  # delete this later
            for i, user_token in enumerate(tqdm(user_channel_chunks)):  # chunk-by-chunk prediction
                user_chunk_special_text = [f"<|audio_{id}|>" for id in user_channel_chunks[i]]
                user_chunk_special_text = "".join(user_chunk_special_text)
                formatted_tokens = "<|reserved_151347|>" + user_chunk_special_text + "<|reserved_151348|>"  # Convert all tokens at once
                user_token_ids = self.glm_tokenizer.encode(formatted_tokens, add_special_tokens=False)
                # user_audio_tokens.extend(user_channel_chunks[i])  # Do not need to store user tokens here, we already have them
                user_token_ids = torch.tensor([user_token_ids]).to(self.device)
                input_ids = torch.cat([input_ids, user_token_ids], dim=1)
                
                # # Append user token to the sequence
                # user_token_id = self.glm_tokenizer.convert_tokens_to_ids(f"<|audio_{user_token}|>")
                # input_ids = torch.cat([input_ids, torch.tensor([[user_token_id]]).to(self.device)], dim=1)

                # print(f"input_ids: {input_ids}")
                # print(f"input_ids.shape: {input_ids.shape}")
                # print("now generating...")
                # exit()
                # Generate a chunk of tokens from the model
                original_input_ids = input_ids
                new_generated_ids = []
                try:
                    while True:
                        outputs = self.glm_model.generate(
                            input_ids=input_ids,
                            max_new_tokens=chunk_size,  # predict one chunk of model tokens
                            temperature=temperature,
                            top_p=top_p,
                            do_sample=True,
                            pad_token_id=self.glm_tokenizer.pad_token_id,
                            eos_token_id=[end_of_audio_offset, special_token_ids["<|reserved_151347|>"], special_token_ids["<|reserved_151348|>"]],  # Only this token will stop generation
                        )
                        
                        # Get the newly generated token
                        new_chunk = outputs[0, -chunk_size:].tolist()
                        if all(token_id < audio_offset and token_id not in [151347, 151348] for token_id in new_chunk):
                            # all the generated tokens are text tokens, append the end_of_text token. The next generation should only contain audio tokens
                            new_chunk.append(151349)

                        new_generated_ids.extend(new_chunk)
                        save_new_generated_ids = new_generated_ids  # delete this later
                        flag, new_generated_ids = check_speech_token_num(new_generated_ids, audio_offset)
                        if flag == False:
                            global_flag = True  # delete this later
                        
                        # Update input_ids for next iteration
                        new_generated_ids_tensor = torch.tensor([new_generated_ids]).to(self.device)
                        input_ids = torch.cat([original_input_ids, new_generated_ids_tensor], dim=1)
                        # input_ids = outputs

                        if flag:
                            break
                except Exception as e:
                    print(f"input_ids error?: {input_ids}")
                    os.makedirs("./error_input_ids", exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filepath = f"./error_input_ids/error_{timestamp}.pt"
                    with open(filepath, "w") as f:
                        f.write(str(input_ids.tolist()))
                    return [], [], "while_true_generation_error"
                    
                speech_tokens = new_generated_ids[-chunk_size:]  # generated speech tokens appear at the end
                # print(f"speech_tokens: {speech_tokens}")
                # if global_flag:
                #     exit()

                # If it's an speech token, add to our collection
                for new_token in speech_tokens:
                    assert new_token >= audio_offset
                    output_model_audio_tokens.append(new_token - audio_offset)
                    
                # add text
                text_token_ids = [token for token in new_generated_ids if token < audio_offset]
                if text_token_ids:
                    texts = self.glm_tokenizer.decode(text_token_ids, skip_special_tokens=True)
                    model_texts += f" {texts}"
                    overall_text["interleaved_text"].append(("model", texts))

        print("===================")
        # print(f"user_audio_tokens: {user_audio_tokens}")
        # print(f"output_model_audio_tokens: {output_model_audio_tokens}")
        print(f"user_audio_tokens len: {len(user_audio_tokens)}")
        print(f"output_model_audio_tokens len: {len(output_model_audio_tokens)}")
        interleaved_token_ids = input_ids[0].detach().cpu().tolist()
        interleaved_tokens = self.glm_tokenizer.convert_ids_to_tokens(interleaved_token_ids)
        print("interleaved_token_ids:")
        print(interleaved_token_ids)
        print("interleaved_tokens:")
        print(interleaved_tokens)

        print(f"output_model_audio_tokens: {output_model_audio_tokens}")
        if len(output_model_audio_tokens) != len(user_audio_tokens):  # Handle the length mismatch case
            # truncate to the shorter one
            print(f"Before length truncate: {len(output_model_audio_tokens)}, {len(user_audio_tokens)}")
            min_length = min(len(output_model_audio_tokens), len(user_audio_tokens))
            output_model_audio_tokens = output_model_audio_tokens[:min_length]
            user_audio_tokens = user_audio_tokens[:min_length]
            print(f"After length truncate: {len(output_model_audio_tokens)}, {len(user_audio_tokens)}")
        # assert len(output_model_audio_tokens) == len(user_audio_tokens), "User and model audio tokens must be of the same length"
        
        print(f"output_model_audio_tokens: {len(output_model_audio_tokens)} tokens")
        # decoded_text = self.glm_tokenizer.decode(output_model_audio_tokens)
        # print(decoded_text)

        # Convert model audio tokens to waveform
        if output_model_audio_tokens:
            # save text
            overall_text["user"] = user_texts.split("<|reserved_151349|>")  # there is not gonna be any user text in this case
            overall_text["model"] = model_texts.split("<|reserved_151349|>")
            text_output_path = stereo_output_path.replace('.wav', '.json')
            with open(text_output_path, 'w') as f:
                json.dump(overall_text, f, ensure_ascii=False, indent=4)

            # save audio
            model_audio_tensor = torch.tensor(output_model_audio_tokens, device=self.device).unsqueeze(0)
            this_uuid = str(uuid.uuid4())
            model_speech, _ = self.audio_decoder.token2wav(
                model_audio_tensor,
                uuid=this_uuid,
                prompt_token=torch.zeros(1, 0, dtype=torch.int64).to(self.device),
                prompt_feat=torch.zeros(1, 0, 80).to(self.device)
            )
            
            # Convert user tokens to waveform for comparison/reference
            user_audio_tensor = torch.tensor(user_audio_tokens, device=self.device).unsqueeze(0)
            user_speech, _ = self.audio_decoder.token2wav(
                user_audio_tensor,
                uuid=str(uuid.uuid4()),
                prompt_token=torch.zeros(1, 0, dtype=torch.int64).to(self.device),
                prompt_feat=torch.zeros(1, 0, 80).to(self.device)
            )
            
            # Save both channels if output dir is provided
            if output_dir:
                # Save model speech as a mono audio file
                torchaudio.save(mono_output_path, model_speech.cpu(), 22050)
                
                # # Save user speech
                # user_output_path = output_path.replace('.wav', '_user.wav')
                # torchaudio.save(user_output_path, user_speech.cpu(), 22050)
                
                # Create a stereo file with both channels
                # stereo_output_path = output_path.replace('.wav', f'_stereo.wav')
                input_user_speech = self._load_mono_audio(input_audio_path)
                self._create_stereo_file(input_user_speech, model_speech.cpu(), stereo_output_path)
                
            return model_speech, user_speech, "normal"
        else:
            print("No model audio tokens were generated")
            return None, None, "no_audio"

    def _load_mono_audio(self, audio_path, sample_rate=22050):
        waveform, original_sample_rate = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if original_sample_rate != sample_rate:
            waveform = torchaudio.functional.resample(waveform, original_sample_rate, sample_rate)
        return waveform
    
    def _create_stereo_file(self, user_audio, model_audio, output_path, sample_rate=22050):
        """
        Create a stereo audio file with user audio in left channel and model audio in right channel.
        Pads the shorter audio to match the length of the longer one.
        """
        # Get lengths
        user_len = user_audio.shape[1]
        model_len = model_audio.shape[1]
        max_len = max(user_len, model_len)
        
        # Pad to same length
        if user_len < max_len:
            user_audio = torch.nn.functional.pad(user_audio, (0, max_len - user_len))
        if model_len < max_len:
            model_audio = torch.nn.functional.pad(model_audio, (0, max_len - model_len))
        
        # Create stereo audio
        stereo_audio = torch.stack([user_audio[0], model_audio[0]], dim=0)
        torchaudio.save(output_path, stereo_audio, sample_rate)
        
        print(f"Saved stereo audio to {output_path}")

    def audio_reconstruction(self, input_audio_path):
        # first extract the tokens from the audio file
        audio_tokens = extract_speech_token(
            self.whisper_model,
            self.feature_extractor,
            [input_audio_path]
        )[0]
        
        # then reconstruct the audio from the tokens
        audio_tensor = torch.tensor(audio_tokens, device=self.device).unsqueeze(0)
        this_uuid = str(uuid.uuid4())  # Generate a unique ID for the audio file
        tts_speech, _ = self.audio_decoder.token2wav(  # TODO: solve the bug: RuntimeError: Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int; but got torch.cuda.FloatTensor instead (while checking arguments for embedding)
            audio_tensor,
            uuid=this_uuid,
            prompt_token=torch.zeros(1, 0, dtype=torch.int64).to(self.device),
            prompt_feat=torch.zeros(1, 0, 80).to(self.device)
        )
        
        return tts_speech


def parse_args():
    parser = argparse.ArgumentParser(
        description="TurnGuide demo: generate assistant audio from a mono user-channel audio file."
    )
    parser.add_argument("--input-audio", required=True, help="Path to the mono user audio file.")
    parser.add_argument("--output-dir", default="./turnguide_demo_output", help="Directory for generated files.")
    parser.add_argument("--temperature", type=float, default=temperature, help="Sampling temperature.")
    parser.add_argument("--p-value", type=float, default=p_value, help="Top-p sampling value.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Path or Hugging Face id for the GLM checkpoint.")
    parser.add_argument(
        "--tokenizer-path",
        default=DEFAULT_TOKENIZER_PATH,
        help="Path or Hugging Face id for the GLM-4-Voice tokenizer.",
    )
    parser.add_argument(
        "--flow-path",
        default=DEFAULT_FLOW_PATH,
        help="Path to the downloaded GLM-4-Voice decoder directory.",
    )
    parser.add_argument("--device", default="cuda", help="Inference device.")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "int4"], help="Model dtype.")
    return parser.parse_args()


def run_demo(args):
    os.makedirs(args.output_dir, exist_ok=True)
    processor = SpeechProcessor(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        flow_path=args.flow_path,
        device=args.device,
        dtype=args.dtype,
    )
    _, _, status = processor.generate_conditional_interleaved_speech_token_only(
        input_audio_path=args.input_audio,
        temperature=args.temperature,
        top_p=args.p_value,
        output_dir=args.output_dir,
    )
    print(f"Generation status: {status}")
    print(f"Assistant audio: {os.path.join(args.output_dir, 'assistant.wav')}")
    print(f"Stereo audio: {os.path.join(args.output_dir, 'stereo_user_left_assistant_right.wav')}")


if __name__ == "__main__":
    run_demo(parse_args())
