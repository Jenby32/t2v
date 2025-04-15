#!/bin/bash
set -e

echo "📦 Starte vollständigen HunyuanVideo Setup..."

# Step 1: Miniconda installieren (wenn nicht vorhanden)
if [ ! -d "$HOME/miniconda3" ]; then
  echo "⬇️  Miniconda wird installiert..."
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
  bash miniconda.sh -b -p $HOME/miniconda3
  source ~/miniconda3/etc/profile.d/conda.sh
  conda init
  source ~/.bashrc
else
  echo "✅ Miniconda bereits installiert"
  source ~/miniconda3/etc/profile.d/conda.sh
fi

# Step 2: Conda-Umgebung vorbereiten
if conda info --envs | grep -q "^hunyuan"; then
  echo "⚠️  Conda-Umgebung 'hunyuan' existiert bereits"
else
  echo "🐍 Erstelle Conda-Umgebung 'hunyuan'..."
  conda create -n hunyuan python=3.10 -y
fi

echo "📂 Aktiviere Umgebung..."
conda activate hunyuan

# Step 3: Hugging Face CLI installieren
echo "🔑 Installiere Hugging Face CLI..."
pip install --upgrade "huggingface_hub[cli]"

# Step 4: PyTorch 2.6.0 mit CUDA 12.4
echo "⚙️  Installiere PyTorch 2.6.0 + CUDA 12.4..."
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Step 5: Transformers & Co
echo "📚 Installiere NLP/Utility Pakete..."
pip install transformers accelerate einops scipy loguru safetensors imageio diffusers

# Step 6: Repo klonen
if [ ! -d "HunyuanVideo" ]; then
  echo "📁 Klone HunyuanVideo Repo..."
  git clone https://github.com/Tencent/HunyuanVideo.git
fi
cd HunyuanVideo

# Step 7: Checkpoint-Verzeichnis vorbereiten
mkdir -p ckpts

# Step 8: Hauptmodell (Text-to-Video) downloaden
echo "🎬 Lade Text-to-Video Modellgewichte (720p)..."
huggingface-cli download Tencent/HunyuanVideo --local-dir ./ckpts/hunyuan-video-t2v-720p --repo-type model

# Step 9: CLIP Textencoder (text_encoder_2)
if [ ! -d "./ckpts/text_encoder_2" ]; then
  echo "🎯 Lade CLIP Textencoder..."
  huggingface-cli download openai/clip-vit-large-patch14 --local-dir ./ckpts/text_encoder_2 --repo-type model
fi

# Step 10: LLaVA / LLM Textencoder (text_encoder)
if [ ! -d "./ckpts/text_encoder" ]; then
  echo "🧠 Lade LLaVA LLM Textencoder & preprocessiere..."
  huggingface-cli download xtuner/llava-llama-3-8b-v1_1-transformers --local-dir ./ckpts/llava
  python hyvideo/utils/preprocess_text_encoder_tokenizer_utils.py \
    --input_dir ./ckpts/llava \
    --output_dir ./ckpts/text_encoder
fi

# Step 11: FlashAttention installieren
echo "⚡ Installiere FlashAttention..."
pip install ninja
pip install flash-attn --no-build-isolation --no-cache-dir

# Step 12: XFormers & xfuser
echo "🧱 Installiere xformers & xfuser..."
pip install xformers xfuser==0.4.0

# Step 13: Check ob alles da ist
echo ""
if [ ! -f "ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt" ]; then
  echo "❌ WICHTIG: Modelldatei fehlt! Prüfe deinen Download!"
  exit 1
fi

echo ""
echo "✅ Setup abgeschlossen!"
echo "👉 Conda-Umgebung 'hunyuan' ist aktiv."
echo "🎬 Starte ein erstes Testvideo mit:"
echo "python sample_video.py --prompt 'A cat walking' --video-size 360 640 --video-length 8 --infer-steps 10 --save-path ./results/test.mp4 --use-cpu-offload"
