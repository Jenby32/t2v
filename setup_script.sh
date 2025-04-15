#!/bin/bash
set -e

echo "📦 Starte vollständigen HunyuanVideo Setup..."

# ===== Step 0: Caches vorher löschen =====
echo "🧹 Leere Cache-Ordner (conda, pip, HF)..."
rm -rf ~/.cache/huggingface
rm -rf ~/.cache/pip
rm -rf ~/miniconda3/pkgs

# ===== Step 1: Miniconda installieren =====
if [ ! -d "$HOME/miniconda3" ]; then
  echo "⬇️  Installiere Miniconda..."
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
  bash miniconda.sh -b -p $HOME/miniconda3
  source ~/miniconda3/etc/profile.d/conda.sh
  conda init
  source ~/.bashrc
else
  echo "✅ Miniconda vorhanden"
  source ~/miniconda3/etc/profile.d/conda.sh
fi

# ===== Step 2: Conda Umgebung =====
if conda info --envs | grep -q "^hunyuan"; then
  echo "⚠️  Conda-Umgebung 'hunyuan' existiert schon"
else
  echo "🐍 Erstelle neue Conda-Umgebung..."
  conda create -n hunyuan python=3.10 -y
fi

echo "📂 Aktiviere Umgebung..."
conda activate hunyuan

# ===== Step 3: Python Pakete =====
echo "📦 Installiere Huggingface CLI + PyTorch + weitere..."
pip install --upgrade "huggingface_hub[cli]"
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install transformers accelerate einops scipy loguru safetensors imageio diffusers

# ===== Step 4: Repo holen =====
if [ ! -d "HunyuanVideo" ]; then
  git clone https://github.com/Tencent/HunyuanVideo.git
fi
cd HunyuanVideo

# ===== Step 5: Checkpoint-Ordner vorbereiten =====
mkdir -p ckpts

# ===== Step 6: Hauptmodell =====
echo "🎬 Lade Hauptmodell (Text-to-Video)..."
huggingface-cli download Tencent/HunyuanVideo \
  --local-dir ./ckpts/hunyuan-video-t2v-720p \
  --repo-type model \
  --resume-download

# === FIX: Falls Ordner doppelt verschachtelt wurde
if [ -d "ckpts/hunyuan-video-t2v-720p/hunyuan-video-t2v-720p" ]; then
  echo "📁 Fix: Verschachtelter Ordner gefunden – korrigiere..."
  mv ckpts/hunyuan-video-t2v-720p/hunyuan-video-t2v-720p/* ckpts/hunyuan-video-t2v-720p/
  rm -rf ckpts/hunyuan-video-t2v-720p/hunyuan-video-t2v-720p
fi

# ===== Step 7: CLIP Textencoder =====
echo "🎯 Lade CLIP Textencoder..."
huggingface-cli download openai/clip-vit-large-patch14 \
  --local-dir ./ckpts/text_encoder_2 \
  --repo-type model \
  --resume-download

# ===== Step 8: LLaVA / LLM Textencoder =====
echo "🧠 Lade LLaVA Textencoder..."
huggingface-cli download xtuner/llava-llama-3-8b-v1_1-transformers \
  --local-dir ./ckpts/llava \
  --repo-type model \
  --resume-download

python hyvideo/utils/preprocess_text_encoder_tokenizer_utils.py \
  --input_dir ./ckpts/llava \
  --output_dir ./ckpts/text_encoder

# ===== Step 9: FlashAttention =====
echo "⚡ Installiere FlashAttention..."
pip install ninja
pip install flash-attn --no-build-isolation --no-cache-dir

# ===== Step 10: xformers + xfuser =====
pip install xformers xfuser==0.4.0

# ===== Step 11: Modellprüfung =====
echo ""
if [ ! -f "ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt" ]; then
  echo "❌ WICHTIG: Hauptmodell fehlt! Prüfe Download in ckpts/hunyuan-video-t2v-720p"
  exit 1
fi

echo ""
echo "✅ Setup erfolgreich abgeschlossen!"
echo "👉 Starte z. B.:"
echo "python sample_video.py --prompt 'A test video' --video-size 360 640 --video-length 8 --infer-steps 10 --save-path ./results/test.mp4 --use-cpu-offload"
