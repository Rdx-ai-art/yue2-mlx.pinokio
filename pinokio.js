module.exports = {
  version: "5.0",
  title: "YUE2 // MLX",
  description: "Mac-optimized YuE2 music generation using native MLX inference. Generate complete songs (melody, chords, vocals, accompaniment) from text prompts. Features an LLM Writing Room for lyric composition via OpenAI-compatible APIs (LM Studio, etc.). Pure Apple Silicon — no PyTorch needed.",
  icon: "icon.png",
  platform: "darwin",
  menu: async (kernel, info) => {
    const hasApp = info.exists("app.py");
    const hasInference = info.exists("yue2_model.py") && info.exists("yue2_vae.py");
    const installed = hasApp && hasInference;
    const running = info.running("start.js");
    const local = info.local("start.js") || {};
    const url = local.url;

    if (!installed) {
      return [
        { icon: "fa-solid fa-plug", text: "Install", href: "install.js", default: true },
      ];
    }

    if (running) {
      if (url) {
        return [
          { icon: "fa-solid fa-rocket", text: "Open Web UI", href: url, default: true },
          { icon: "fa-solid fa-terminal", text: "Terminal", href: "start.js" },
          { icon: "fa-solid fa-rotate", text: "Update", href: "update.js" },
          { icon: "fa-solid fa-broom", text: "Reset", href: "reset.js" },
        ];
      }
      return [
        { icon: "fa-solid fa-circle-notch fa-spin", text: "Starting", href: "start.js", default: true },
        { icon: "fa-solid fa-rotate", text: "Update", href: "update.js" },
        { icon: "fa-solid fa-broom", text: "Reset", href: "reset.js" },
      ];
    }

    return [
      { icon: "fa-solid fa-power-off", text: "Start", href: "start.js", default: true },
      { icon: "fa-solid fa-rotate", text: "Update", href: "update.js" },
      { icon: "fa-solid fa-broom", text: "Reset", href: "reset.js" },
    ];
  }
};
