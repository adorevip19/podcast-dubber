# Podcast Dubber

YouTube 视频自动中文配音服务。提供一个 YouTube 链接，自动完成音频提取、语音识别、中文翻译、语音合成和时间对齐，输出配音后的 MP3 文件。

## 处理流程

1. **音频提取** \u2013 yt-dlp 下载 YouTube 视频音频 (128kbps MP3)
2. **语音识别** \u2013 OpenAI Whisper API 转录，获取带时间戳的 segments
3. **中文翻译** \u2013 GPT-4o 批量翻译为简体中文（每批 30 段）
4. **语音合成** \u2013 OpenAI TTS API (tts-1, nova 音色) 逐段生成中文语音
5. **音频拼接** \u2013 pydub 按原始时间戳对齐拼接，短了补静音，长了加速（最多 1.2x）或截断

## 本地运行

### 前置条件

- Python 3.10+
- ffmpeg (\`brew install ffmpeg\` / \`apt install ffmpeg\`)
- OpenAI API Key

### 步骤

\`\`\`bash
git clone https://github.com/adorevip19/podcast-dubber.git
cd podcast-dubber
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env，填入你的 OPENAI_API_KEY
export $(cat .env | xargs)
uvicorn main:app --reload --port 8000
\`\`\`

### 使用 API

\`\`\`bash
# 提交任务
curl -X POST http://localhost:8000/dub \\
  -H "Content-Type: application/json" \\
  -d '{"youtube_url": "https://www.youtube.com/watch?v=VIDEO_ID"}'

# 查询状态
curl http://localhost:8000/dub/{job_id}

# 完成后下载
curl -O http://localhost:8000/dub/{job_id}/download
\`\`\`

## 部署到 Railway

1. 登录 [Railway](https://railway.app)，点击 **New Project \u2192 Deploy from GitHub repo**
2. 选择 \`podcast-dubber\` 仓库
3. 在 Variables 中添加 \`OPENAI_API_KEY\`
4. Railway 会自动检测 railway.toml 并部署

## API 文档

部署后访问 \`/docs\` 查看 Swagger 交互式文档。
