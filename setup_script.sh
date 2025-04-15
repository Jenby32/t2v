#!/bin/bash
set -e

echo "📦 Starte vollständigen HunyuanVideo Setup..."

### 0. Caches löschen (wird später mehrfach wiederholt!)
clear_caches() {
  echo "🧹 Leere Cache (pip, conda, HF)..."
  rm -rf ~/.cache/huggingface
  rm -rf ~/.cache/pip
  rm -rf ~/.conda/pkgs
}
clear_caches

### 1. Miniconda Setup
if [ ! -d "$HOME/miniconda3" ]; then
  echo "⬇️  Installiere Miniconda..."
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
  bash miniconda.sh -b -p $HOME/miniconda3
  eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
  conda init
  source ~/.bashrc
else
  echo "✅ Miniconda vorhanden"
  if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  else
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
  fi
fi

### 2. Conda Env erstellen + aktivieren
if conda info --envs | grep -q "^hunyuan"; then
  echo "⚠️  Conda-Env 'hunyuan' existiert bereits"
else
  echo "🐍 Erstelle Conda-Env 'hunyuan'..."
  conda create -n hunyuan python=3.10 -y
fi
conda activate hunyuan

### 3. Essentials
echo "🔧 Installiere Grundpakete..."
pip install --upgrade pip
pip install "huggingface_hub[cli]" ninja

### 4. PyTorch mit CUDA 12.4
echo "⚙️  Installiere PyTorch 2.6.0 + CUDA 12.4..."
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

### 5. NLP / Utility Libs
echo "🧠 Installiere Transformers etc..."
pip install transformers accelerate einops scipy loguru safetensors imageio diffusers

clear_caches

### 6. Repo klonen
if [ ! -d HunyuanVideo ]; then
  echo "📁 Klone HunyuanVideo Repo..."
  git clone https://github.com/Tencent/HunyuanVideo.git
fi
cd HunyuanVideo
mkdir -p ckpts

### 7. Modellgewichte laden (wie in der HF-Doku)
echo "📥 Lade Hauptmodell (Text2Video 720p)..."
huggingface-cli download tencent/HunyuanVideo --local-dir ./ckpts

clear_caches

### 8. CLIP Encoder
echo "🎯 Lade CLIP Textencoder..."
huggingface-cli download openai/clip-vit-large-patch14 --local-dir ./ckpts/text_encoder_2 --repo-type model

clear_caches

### 9. LLaVA Encoder + Preprocessing
echo "🧠 Lade LLaVA Textencoder (LLaMA 3) und preprocessiere..."
huggingface-cli download xtuner/llava-llama-3-8b-v1_1-transformers --local-dir ./ckpts/llava
python hyvideo/utils/preprocess_text_encoder_tokenizer_utils.py \
  --input_dir ./ckpts/llava \
  --output_dir ./ckpts/text_encoder

clear_caches

### 10. FlashAttention + xformers
echo "⚡ Installiere FlashAttention & xformers..."
pip install flash-attn --no-build-isolation --no-cache-dir
pip install xformers xfuser==0.4.0

clear_caches

### 11. Check
echo ""
if [ ! -f "ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt" ]; then
  echo "❌ WICHTIG: Modelldatei fehlt! Prüfe deinen Download oder Verzeichnisstruktur!"
  exit 1
fi

echo ""
echo "✅ Setup abgeschlossen!"
echo "👉 Umgebung: 'hunyuan' ist aktiv"
echo "🎬 Testvideo generieren mit z. B.:"
echo ""
echo "python sample_video.py --prompt 'A cat walking' \\"
echo "  --video-size 360 640 --video-length 8 --infer-steps 10 \\"
echo "  --save-path ./results/test.mp4 --use-cpu-offload"
