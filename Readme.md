# RAG System Documentation

ระบบ Retrieval-Augmented Generation (RAG) สำหรับการค้นหาและตอบคำถามจากเอกสาร PDF โดยใช้ OpenSearch, Ollama และ Streamlit

## 📋 สารบัญ

- [ภาพรวมระบบ](#ภาพรวมระบบ)
- [สถาปัตยกรรม](#สถาปัตยกรรม)
- [ความต้องการของระบบ](#ความต้องการของระบบ)
- [การติดตั้ง](#การติดตั้ง)
- [การใช้งาน](#การใช้งาน)
- [โครงสร้างโปรเจค](#โครงสร้างโปรเจค)
- [การกำหนดค่า](#การกำหนดค่า)
- [การแก้ไขปัญหา](#การแก้ไขปัญหา)

## 📹 วีดีโอสาธิตการติดตั้ง

🔗 [**ดูวีดีโอสาธิตการติดตั้งแบบเต็ม:**  ](https://minddatatech.com/media/AdvanceRAG/Advanced_RAG-2025-10-19.mp4)

## 🎯 ภาพรวมระบบ

ระบบนี้ประกอบด้วย 6 services หลัก ที่ทำงานร่วมกันเพื่อสร้างระบบ RAG ที่สมบูรณ์:

- **OpenSearch**: ฐานข้อมูลสำหรับจัดเก็บและค้นหา embeddings
- **Embedding Service**: ประมวลผลเอกสาร PDF และสร้าง embeddings
- **Search API**: API สำหรับการค้นหาข้อมูล
- **Backend**: ประมวลผลคำถามและสร้างคำตอบ
- **Frontend**: UI สำหรับผู้ใช้งาน (Streamlit)
- **Ollama**: LLM สำหรับการสร้างคำตอบ (qwen2.5:0.5b)

## 🏗️ สถาปัตยกรรม

```
┌─────────────┐
│  Frontend   │ :8501 (Streamlit)
└──────┬──────┘
       │
       ↓
┌─────────────┐
│   Backend   │ :8006
└──────┬──────┘
       │
       ├──────→ ┌──────────────┐
       │        │  Search API  │ :8005
       │        └──────┬───────┘
       │               │
       │               ↓
       │        ┌──────────────┐
       │        │  OpenSearch  │ :9200
       │        └──────────────┘
       │               ↑
       │               │
       │        ┌──────────────┐
       │        │  Embedding   │
       │        └──────────────┘
       │
       └──────→ ┌──────────────┐
                │    Ollama    │ :11434
                └──────────────┘
```

## 💻 ความต้องการของระบบ

- Docker Engine 20.10+
- Docker Compose 1.29+
- RAM อย่างน้อย 8GB (แนะนำ 16GB)
- พื้นที่ว่างในดิสก์ อย่างน้อย 10GB

## 🚀 การติดตั้ง

### 1. Clone โปรเจค

```bash
git clone <repository-url>
cd <project-directory>
```

### 2. สร้างโครงสร้างโฟลเดอร์

```bash
mkdir -p embedding/pdf_corpus
mkdir -p embedding/index
```

### 3. วางไฟล์ PDF

ใส่ไฟล์ PDF ที่ต้องการให้ระบบประมวลผลในโฟลเดอร์:
```
embedding/pdf_corpus/
```

### 4. สร้างและรัน Services

```bash
# สร้างและรัน containers ทั้งหมด
docker-compose up -d

# ดู logs
docker-compose logs -f

# ดู logs ของ service เฉพาะ
docker-compose logs -f frontend_container
```

## 📝 การใช้งาน

### เข้าถึง Frontend

เปิดเว็บเบราว์เซอร์และไปที่:
```
http://localhost:8501
```

### API Endpoints

**Search API** (พอร์ต 8005):
```bash
# Health check
curl http://localhost:8005/health

# ค้นหา
curl -X POST http://localhost:8005/search \
  -H "Content-Type: application/json" \
  -d '{"query": "your search query"}'
```

**Backend API** (พอร์ต 8006):
```bash
# Health check
curl http://localhost:8006/health

# Query
curl -X POST http://localhost:8006/query \
  -H "Content-Type: application/json" \
  -d '{"question": "your question"}'
```

**OpenSearch** (พอร์ต 9200):
```bash
# Cluster health
curl http://localhost:9200/_cluster/health

# List indices
curl http://localhost:9200/_cat/indices?v
```

**Ollama** (พอร์ต 11434):
```bash
# Check status
curl http://localhost:11434
```

## 📁 โครงสร้างโปรเจค

```
.
├── docker-compose.yml
├── opensearch_db/
│   └── Dockerfile
├── embedding/
│   ├── Dockerfile
│   ├── pdf_corpus/         # วางไฟล์ PDF ที่นี่
│   └── index/              # index files (auto-generated)
├── search_api/
│   └── Dockerfile
├── backend/
│   └── Dockerfile
├── frontend/
│   └── Dockerfile
└── ollama_llm/
    └── Dockerfile
```

## ⚙️ การกำหนดค่า

### Environment Variables

#### OpenSearch
```yaml
OPENSEARCH_JAVA_OPTS: -Xms2g -Xmx2g  # Java heap size
plugins.security.disabled: true      # ปิด security (development only)
```

#### Embedding Service
```yaml
WATCH_INTERVAL: 60                                    # ตรวจสอบไฟล์ใหม่ทุก 60 วินาที
OPENSEARCH_ENDPOINT: http://opensearch_container:9200
```

#### Backend
```yaml
SEARCH_API_URL: http://search_api_container:8005
OLLAMA_URL: http://ollama_container:11434
```

#### Ollama
```yaml
MODEL_NAME: qwen2.5:0.5b  # เปลี่ยนได้ตามต้องการ
```

### Ports

| Service | Port | Description |
|---------|------|-------------|
| OpenSearch | 9200, 9300 | Database & Transport |
| Search API | 8005 | Search service |
| Backend | 8006 | Main API |
| Frontend | 8501 | Web UI |
| Ollama | 11434 | LLM service |

## 🔧 การจัดการ

### คำสั่งพื้นฐาน

```bash
# เริ่มทุก services
docker-compose up -d

# หยุดทุก services
docker-compose down

# หยุดและลบ volumes
docker-compose down -v

# Restart service เฉพาะ
docker-compose restart <service_name>

# ดูสถานะ
docker-compose ps

# ดู logs
docker-compose logs -f <service_name>

# Rebuild service
docker-compose up -d --build <service_name>
```

### การเพิ่มเอกสารใหม่

1. วางไฟล์ PDF ใหม่ในโฟลเดอร์ `embedding/pdf_corpus/`
2. Embedding service จะตรวจสอบและประมวลผลอัตโนมัติภายใน 60 วินาที
3. ตรวจสอบ logs: `docker-compose logs -f embedding_container`

### Backup ข้อมูล

OpenSearch data และ Ollama models ถูกเก็บใน Docker volumes:
```bash
# แสดง volumes
docker volume ls | grep rag

# Backup volume
docker run --rm -v <volume_name>:/data -v $(pwd):/backup \
  alpine tar czf /backup/backup.tar.gz /data
```

## 🐛 การแก้ไขปัญหา

### OpenSearch ไม่ทำงาน

```bash
# ตรวจสอบ logs
docker-compose logs opensearch_container

# ตรวจสอบ cluster health
curl http://localhost:9200/_cluster/health

# Restart service
docker-compose restart opensearch_container
```

### Embedding Service ประมวลผลไม่สำเร็จ

```bash
# ตรวจสอบว่ามี PDF ในโฟลเดอร์
ls -la embedding/pdf_corpus/

# ตรวจสอบ logs
docker-compose logs -f embedding_container

# Restart service
docker-compose restart embedding_container
```

### Ollama ไม่ตอบสนอง

```bash
# ตรวจสอบว่า model ถูกโหลดแล้ว
docker exec ollama_container ollama list

# Pull model ใหม่
docker exec ollama_container ollama pull qwen2.5:0.5b

# Restart
docker-compose restart ollama_container
```

### Health Check ล้มเหลว

```bash
# ตรวจสอบสถานะ health
docker-compose ps

# ตรวจสอบ health check แต่ละ service
docker inspect <container_name> | grep -A 20 Health
```

### ปัญหา Memory

หาก OpenSearch หรือ Ollama ใช้ memory มากเกินไป:

```yaml
# ลด heap size ของ OpenSearch
OPENSEARCH_JAVA_OPTS: -Xms1g -Xmx1g

# หรือใช้ model ที่เล็กกว่า
MODEL_NAME: qwen2.5:0.5b  # หรือ tinyllama
```

## 📊 Monitoring

### ตรวจสอบ Resource Usage

```bash
# ดู CPU, Memory ของทุก containers
docker stats

# ดูเฉพาะ container
docker stats <container_name>
```

### ตรวจสอบ Logs

```bash
# Real-time logs
docker-compose logs -f

# Logs ย้อนหลัง 100 บรรทัด
docker-compose logs --tail=100

# Logs ของช่วงเวลา
docker-compose logs --since 30m
```

## 🔒 Security Notes

⚠️ **สำคัญ**: Configuration ปัจจุบันเหมาะสำหรับ development เท่านั้น

สำหรับ production:
1. เปิด OpenSearch security (`plugins.security.disabled: false`)
2. ใช้ authentication สำหรับทุก services
3. จำกัด CORS policies
4. ใช้ HTTPS
5. จำกัดการเข้าถึง ports
6. ตั้งค่า resource limits

## 📚 Additional Resources

- [OpenSearch Documentation](https://opensearch.org/docs/)
- [Ollama Documentation](https://github.com/ollama/ollama)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Streamlit Documentation](https://docs.streamlit.io/)

## 📄 License

[ระบุ license ของโปรเจค]

## 👥 Contributors

[ระบุผู้พัฒนา]

---

**หมายเหตุ**: อย่าลืมปรับแต่งค่า configuration ให้เหมาะสมกับ environment ของคุณ