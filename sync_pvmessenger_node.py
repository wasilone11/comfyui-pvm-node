import time
import uuid
import tempfile
import os
import shutil
import wave
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

import requests
import pandas as pd
from moviepy import VideoFileClip
from sync import Sync
from sync.common import Video, Audio, GenerationOptions
from sync.core.api_error import ApiError
from elevenlabs.client import ElevenLabs


def _convert_github_blob(url: str) -> str:
    if "github.com" in url and "/blob/" in url:
        parts = url.split("github.com/")[1]
        raw = parts.replace("blob/", "")
        return f"https://raw.githubusercontent.com/{raw}"
    return url


def search_voice(name: str, api_key: str) -> Optional[str]:
    try:
        resp = requests.get(
            "https://api.elevenlabs.io/v2/voices",
            headers={"xi-api-key": api_key},
            params={"search": name, "include_total_count": True},
            timeout=10
        )
        resp.raise_for_status()
        data = resp.json()
        for v in data.get("voices", []):
            if v.get("name") == name:
                return v.get("voice_id")
    except Exception:
        pass
    return None


def delete_voice(voice_id: str, api_key: str) -> None:
    try:
        resp = requests.delete(
            f"https://api.elevenlabs.io/v1/voices/{voice_id}",
            headers={"xi-api-key": api_key},
            timeout=10
        )
        resp.raise_for_status()
        print(f"Deleted voice {voice_id} (status {resp.json().get('status')}).")
    except Exception as e:
        print(f"Failed to delete voice {voice_id}: {e}")


@dataclass
class PVMessengerArgs:
    sync_api_key: str
    eleven_api_key: str
    input_csv_path: str
    output_csv_path: str
    poll_interval: int = 10
    tmp_dir: Path = field(
        default_factory=lambda: Path(tempfile.gettempdir()) / "sync_pvmessenger"
    )


class SyncPVMessengerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sync_api_key": ("STRING", {"default": ""}),
                "eleven_api_key": ("STRING", {"default": ""}),
                "input_csv_path": ("STRING", {"default": "input.csv"}),
                "output_csv_path": ("STRING", {"default": "output.csv"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_csv_path",)
    FUNCTION = "run_pvmessenger"
    CATEGORY = "Sync.so"
    OUTPUT_NODE = True

    def run_pvmessenger(
        self,
        sync_api_key: str,
        eleven_api_key: str,
        input_csv_path: str,
        output_csv_path: str,
    ) -> str:
        args = PVMessengerArgs(
            sync_api_key=sync_api_key,
            eleven_api_key=eleven_api_key,
            input_csv_path=input_csv_path,
            output_csv_path=output_csv_path,
        )
        args.tmp_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(input_csv_path)
        if df.empty:
            print(f"Input CSV {input_csv_path} is empty or invalid.")
            return ""

        for col in ["voice_id", "lipsync_jobID"]:
            if col not in df.columns:
                df[col] = ""
        df["voice_id"] = df["voice_id"].fillna("").astype(str).str.strip()

        voice_name = "my_voice_clone"
        if not df.at[0, "voice_id"]:
            print("Searching for existing voice clone…")
            existing_id = search_voice(voice_name, eleven_api_key)
            if existing_id:
                voice_id = existing_id
                print(f"Using existing voice: {voice_id}")
            else:
                print("Cloning new voice from audio sample…")
                temp_video = self.prepare_video(df.at[0, "video"], args.tmp_dir)
                temp_audio = self.extract_audio(temp_video)
                try:
                    voice_id = self.clone_voice(voice_name, temp_audio, eleven_api_key)
                except ApiError as e:
                    if e.status_code == 400:
                        old = search_voice(voice_name, eleven_api_key)
                        if old:
                            delete_voice(old, eleven_api_key)
                            voice_id = self.clone_voice(voice_name, temp_audio, eleven_api_key)
                        else:
                            raise
                    else:
                        raise
                finally:
                    for f in (temp_video, temp_audio):
                        try: os.remove(f)
                        except: pass
                print(f"Cloned voice: {voice_id}")
            df.loc[df["voice_id"]=="", "voice_id"] = voice_id
        else:
            voice_id = df.loc[df["voice_id"]!="", "voice_id"].iloc[0]
            df.loc[df["voice_id"]=="", "voice_id"] = voice_id

        client = Sync(api_key=sync_api_key).generations
        jobs: List[tuple] = []

        for idx, row in df.iterrows():
            print(f"Processing row {idx+1}/{len(df)}")

            try:
                audio_bytes = self.generate_tts(
                    text=row.get("text", ""),
                    voice_id=voice_id,
                    eleven_api_key=eleven_api_key,
                    tts_model=row.get("tts_model", "eleven_multilingual_v2"),
                )
            except Exception as e:
                print(f"TTS error: {e}")
                df.at[idx, "audio"] = ""
                continue

            tmp_mp3 = args.tmp_dir / f"tts_{uuid.uuid4().hex[:8]}.mp3"
            with open(tmp_mp3, "wb") as f: f.write(audio_bytes)
            aud_url = self.upload_to_uguu(tmp_mp3); os.remove(tmp_mp3)
            df.at[idx, "audio"] = aud_url or ""

            start = float(row.get("segment_start", -1))
            end = float(row.get("segment_end", -1))
            segs = [[start, end]] if 0 <= start < end else None

            video_path = self.prepare_video(row["video"], args.tmp_dir)
            video_url = self.upload_to_uguu(Path(video_path))
            if not video_url:
                print(f" Failed to upload video for row {idx}")
                df.at[idx, "lipsync_jobID"] = "FAILED"
                continue

            video_obj = Video(url=video_url, segments_secs=segs)

            job_id = None
            try:
                res = client.create(
                    input=[video_obj, Audio(url=aud_url)],
                    model=row.get("lipsync_model", "lipsync-2"),
                    options=GenerationOptions(sync_mode=row.get("sync_mode", "bounce")),
                )
                job_id = res.id
                print(f"Job submitted: {job_id}")
            except ApiError as e:
                print(f"Sync error {e.status_code}: {e.body}")
                try:
                    err = json.loads(e.body)
                    job_id = err.get("id")
                except:
                    pass

            df.at[idx, "lipsync_jobID"] = job_id or ""
            if job_id: jobs.append((idx, job_id))
            df.to_csv(output_csv_path, index=False)

        # Save job_ids to JSON
        job_ids = [j for _, j in jobs]
        if job_ids:
            job_json_path = Path(output_csv_path).with_name("job_ids.json")
            with open(job_json_path, "w") as f:
                json.dump(job_ids, f, indent=2)
            print(f"Saved job_ids to {job_json_path}")

        start_ts = time.time()
        pending = {j for _, j in jobs}
        while pending and time.time() - start_ts < 3600:
            time.sleep(args.poll_interval)
            for idx, jid in list(jobs):
                if jid not in pending: continue
                try:
                    st = client.get(jid)
                    if st.status == "COMPLETED":
                        df.at[idx, "output_url"] = st.output_url
                        pending.remove(jid)
                    elif st.status == "FAILED":
                        df.at[idx, "output_url"] = "FAILED"
                        pending.remove(jid)
                except Exception as e:
                    print(f"Check error {e}")
            df.to_csv(output_csv_path, index=False)

        print(f" Done: {output_csv_path}")
        shutil.rmtree(args.tmp_dir, ignore_errors=True)
        return output_csv_path

    def prepare_video(self, ref: str, dest: Path) -> str:
        r = _convert_github_blob(ref)
        if r.startswith(("http://", "https://")):
            return self.download(r, dest)
        dst = dest / Path(r).name
        shutil.copy(r, dst)
        return str(dst)

    def extract_audio(self, video: str) -> str:
        clip = VideoFileClip(video)
        wav = Path(video).with_suffix(".wav")
        clip.audio.write_audiofile(str(wav), logger=None)
        clip.close()
        return str(wav)

    def clone_voice(self, name: str, reference_audio: str, key: str) -> str:
        files = {"files": open(reference_audio, "rb")}
        payload = {"name": name}
        try:
            response = requests.post(
                f"https://api.elevenlabs.io/v1/voices/add",
                files=files,
                data=payload,
                headers={"xi-api-key": key},
                timeout=30,
            )
            response.raise_for_status()
            voice_id = response.json().get("voice_id")
            if not voice_id:
                print(response.json())
                raise ValueError("Voice cloning response did not contain a voice_id")
            print(f"Successfully cloned voice with ID: {voice_id}")
            return voice_id
        except Exception as e:
            print(f"Voice cloning failed: {e}")
            raise ValueError(f"Voice cloning failed: {e}")
        finally:
            for fobj in files.values():
                try:
                    fobj.close()
                except:
                    pass

    def generate_tts(
        self,
        text: str,
        voice_id: str,
        eleven_api_key: str,
        tts_model: str = "eleven_multilingual_v2",
    ) -> bytes:
        cl = ElevenLabs(api_key=eleven_api_key)
        stream = cl.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            model_id=tts_model,
            output_format="mp3_44100_128",
            optimize_streaming_latency=0,
        )
        return b"".join(stream) if not isinstance(stream, (bytes, bytearray)) else stream

    def upload_to_uguu(self, path: Path) -> Optional[str]:
        try:
            with open(path, "rb") as f:
                r = requests.post("https://uguu.se/upload", files=[("files[]", f)], timeout=30)
            j = r.json()
            if j.get("success") and j.get("files"):
                return j["files"][0]["url"]
        except Exception as e:
            print(f"Upload error: {e}")
        return None

    def download(self, url: str, dest: Path) -> str:
        out = dest / Path(url).name
        r = requests.get(url, stream=True, timeout=30)
        r.raise_for_status()
        with open(out, "wb") as f:
            for ch in r.iter_content(1024 * 1024):
                f.write(ch)
        return str(out)


NODE_CLASS_MAPPINGS = {"SyncPVMessengerNode": SyncPVMessengerNode}
NODE_DISPLAY_NAME_MAPPINGS = {"SyncPVMessengerNode": "Sync.so Personalized Video Messenger"}

print(" Sync.so Personalized Video Messenger node loaded.")
