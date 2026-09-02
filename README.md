# Yuz Tanima Sistemi

Fotoğraf, video ve canlı kamera görüntülerinde çoklu yüz tespiti ve tanıma yapan;
bilinen/anonim kimlikleri, process geçmişini ve görünme zamanlarını yöneten tam
Docker tabanlı bir uygulamadır. YOLOv8-Face yüz tespiti, ArcFace R50 yüz vektörü,
PostgreSQL, Qdrant ve MinIO birlikte kullanılır.

## Hizli baslangic

Gereksinimler:

- Git ve [Git LFS](https://git-lfs.com/)
- Docker Desktop ve Docker Compose
- NVIDIA ekran kartı, güncel NVIDIA sürücüsü ve Docker GPU desteği

```powershell
git clone https://github.com/efe-ince23/Yuz-Tanima-Sistemi.git
cd Yuz-Tanima-Sistemi
git lfs pull
Copy-Item .env.example .env
docker compose up -d --build
docker compose ps
```

Linux/macOS üzerinde `.env` oluşturmak için `cp .env.example .env` kullanılabilir.
İlk çalıştırmadan önce `.env` içindeki `change_this_*` değerlerini güçlü ve
benzersiz parolalarla değiştirmek gerekir. Backend açılırken Alembic migration'ları
otomatik uygulanır; PostgreSQL, Qdrant ve MinIO verileri Docker volume'larında
kalıcı tutulur.

## Servisler

Backend: http://localhost:8000

Frontend: http://localhost:3000

API dokumani: http://localhost:8000/docs

pgAdmin: http://localhost:5050

Qdrant Dashboard: http://localhost:6333/dashboard

MinIO Console: http://localhost:9001

PostgreSQL: `localhost:5432`

Yerel geliştirme kullanıcı bilgileri `.env` dosyasında tutulur ve bu dosya Git'e
eklenmez. İlk yönetici hesabı `.env` içindeki `ADMIN_INITIAL_*` değerleriyle
oluşturulur.

## Mimarinin ozeti

- **Frontend:** React, TypeScript ve Vite tabanlı yönetici/kullanıcı arayüzü.
- **Backend:** FastAPI; kimlik doğrulama, fotoğraf/video analizi, geçmiş ve arama API'leri.
- **Yüz tespiti:** YOLOv8-Face ile karedeki tüm yüzlerin bounding box tespiti.
- **Yüz tanıma:** ArcFace R50 ile 512 boyutlu embedding ve eşik tabanlı eşleştirme.
- **PostgreSQL:** Kullanıcı, kişi, face ID, process, olay, video track ve zaman kayıtları.
- **Qdrant:** Yüz embedding'lerinin hızlı benzerlik araması.
- **MinIO:** Referans fotoğrafları, anonim yüz görselleri ve videolar.
- **Docker Compose:** Tüm servislerin tek komutla kurulması ve kalıcı volume yönetimi.

`models/` altındaki ONNX dosyaları Git LFS ile sürümlenir. Klon sonrasında dosyalar
LFS pointer olarak görünüyorsa `git lfs install` ve `git lfs pull` çalıştırılmalıdır.

## Docker yapilandirmasi

`docker compose up -d --build` komutu PostgreSQL, Qdrant, MinIO, API, frontend ve
pgAdmin servislerini tek seferde baslatir. Backend acilisinda Alembic migration'lari
otomatik uygulanir; ayrica manuel bir komut gerekmez. Servislerin durumu
`docker compose ps` ile kontrol edilebilir.

Docker ayarlari `.env` dosyasindan degistirilebilir. Kullanilabilir degiskenler
ve guvenli ornek degerler `.env.example` dosyasinda bulunur:

- `API_HOST_PORT`: API'nin bilgisayardan erisilen portu.
- `API_PORT`: API'nin backend container'i icindeki portu.
- `FRONTEND_PORT`, `POSTGRES_PORT`, `PGADMIN_PORT`: Diger servis portlari.
- `DATA_ROOT`: Yuz verilerinin container icindeki kalici veri yolu.
- `FACE_DATA_PATH`: Yuz verilerinin bilgisayardaki kalici klasoru.
- `FACE_MODEL_NAME`: InsightFace model paketi.
- `FACE_MATCH_THRESHOLD`, `FACE_ANONYMOUS_MATCH_THRESHOLD`: Tanima esikleri.
- `FACE_DUPLICATE_THRESHOLD`: Ayni kisinin tekrar kaydini engelleme esigi.
- `ANONYMOUS_FACE_MAX_SAMPLES`: Anonim kimlik basina azami yuz ornegi.
- `ANONYMOUS_FACE_DUPLICATE_THRESHOLD`: Tekrarlanan anonim ornek esigi.
- `PRIMARY_FACE_DOMINANCE_RATIO`: Ana yuz secim orani.
- `VIDEO_SAMPLE_FPS`: Video yuz tespiti icin saniyede incelenen kare sayisi; varsayilan 6.
- `VIDEO_RECOGNITION_SAMPLES_PER_WINDOW`: Her zaman penceresinde ArcFace'e verilen en kaliteli yuz sayisi.
- `VIDEO_RECOGNITION_WINDOW_SECONDS`: Zamansal tanima karar penceresinin suresi.
- `VIDEO_TRACKER_ENGINE`: `custom` (varsayilan) veya deneysel `bytetrack` takip motoru.
- `VIDEO_BYTETRACK_ACTIVATION_THRESHOLD`, `VIDEO_BYTETRACK_MATCHING_THRESHOLD`: ByteTrack etkinlestirme ve eslestirme ayarlari.
- `QDRANT_HTTP_PORT`, `QDRANT_GRPC_PORT`: Qdrant servis portlari.
- `QDRANT_URL`, `QDRANT_COLLECTION`: Vektor servisi ve collection adi.
- `QDRANT_PREFER_GRPC`: Vektor aramalarinda daha dusuk gecikmeli gRPC iletisimini tercih eder.
- `QDRANT_INDEXING_THRESHOLD_KB`, `QDRANT_FULL_SCAN_THRESHOLD_KB`: HNSW indeksleme ve tam tarama esikleri.
- `FACE_VECTOR_SIZE`: InsightFace yuz vektoru boyutu; varsayilan 512.
- `MINIO_API_PORT`, `MINIO_CONSOLE_PORT`: MinIO S3 API ve web paneli portlari.
- `MINIO_ENDPOINT`, `MINIO_BUCKET`: MinIO servis adresi ve fotograf bucket'i.
- `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`: Backend'in MinIO erisim bilgileri.
- `OBJECT_STORAGE_BACKEND`: `minio` veya gelistirme icin `local` depolama secimi.

PostgreSQL kayitlari `yuz-tanima-postgres-data`, Qdrant verileri
`yuz-tanima-qdrant-data`, MinIO fotograflari `yuz-tanima-minio-data`, model
dosyalari `yuz-tanima-face-models` ve pgAdmin ayarlari
`yuz-tanima-pgadmin-data` adli Docker volume'larinda saklanir. Yedek process
loglari `FACE_DATA_PATH` klasorunde kalir. Bu veriler container yeniden
olusturuldugunda veya yeniden baslatildiginda kaybolmaz.

Gercek JPEG fotograflari MinIO'daki ozel `face-images` bucket'inda nesne olarak
saklanir. PostgreSQL yalnizca `persons/{person_id}/{uuid}.jpg` veya
`anonymous/{face_id}/{uuid}.jpg` bicimindeki object key degerini tutar. Arayuzun
`/media/...` adresleri degismez; backend bu adresleri MinIO'dan okuyup sunar.
Backend acilisinda PostgreSQL'de referans verilen eski yerel fotograflar MinIO'ya
eksikse otomatik ve idempotent olarak aktarilir. Eski `FACE_DATA_PATH` kopyasi
geri donus ve process fallback loglari icin korunabilir.

Yuz vektorlerinin birincil benzerlik aramasi Qdrant'taki `face_embeddings`
collection'i uzerinden yapilir. Her fotograf ornegi ayri bir Qdrant point'i,
ayni kisiye ait ornekler ise ortak `faceId` payload'i ile saklanir. Kisi adi,
metadata, process ve gecmis kayitlari PostgreSQL'de kalir. Qdrant gecici olarak
erisilemezse API mevcut PostgreSQL/pgvector verilerini kullanarak tanimaya devam
eder. Backend her acilista PostgreSQL kaynak verileriyle Qdrant collection'ini
yeniden uzlastirir.

Qdrant kayitlari `yuz-tanima-qdrant-data` adli Docker volume'unda kalici olarak
saklanir. Dashboard'da point payload'lari ve 512 boyutlu vektorler incelenebilir.
MinIO Console'da `face-images` bucket'i altindaki gercek fotograflar incelenebilir.
Collection optimizer ayarlari backend acilisinda idempotent olarak uygulanir;
varsayilan 1000 KB esigi, orta boy veri setlerinde segmentlerin HNSW indeksine
alinmasini ve buyuyen kayitlarda brute-force aramaya bagimli kalinmamasini saglar.

Varsayilan Compose yapisi NVIDIA GPU ister ve CUDA/TensorRT kullanir. Aktif
saglayicilar `GET /health` cevabindaki `execution_providers` alanindan
dogrulanabilir.

Public API, Swagger dokumaninda yalnizca su islemleri sunar:

- `POST /faces/recognize`
- `POST /faces/enroll`
- `GET /faces/{faceId}`
- `DELETE /faces/{faceId}`
- `GET /faces/{faceId}/history`
- `GET /processes/{processId}`

Frontend'in kullandigi `/api` uyumluluk endpoint'leri calismaya devam eder,
ancak public OpenAPI dokumaninda gosterilmez.

Goruntu girislerinde JPEG, PNG ve WebP desteklenir. MIME tipi ile gercek dosya
imzasi birlikte dogrulanir; bos, bozuk veya uyumsuz dosyalar anlamli HTTP hata
kodlariyla reddedilir. Tanima isteginde yuz bulunamazsa HTTP 200 ile `no_face`
durumu doner ve bu sonuc tanima istatistiklerine eklenmez.

`POST /faces/recognize`, goruntudeki tum yuzleri soldan saga bagimsiz olarak
isler. Public cevap her goruntu icin `processId`, `detectedFaceCount` ve `faces`
alanlarini dondurur. Her yuz sonucu `faceId`, `status`, `name`, `metadata`,
`boundingBox` ve `confidence` alanlarini icerir:

- `known`: Sisteme isim ve metadata ile kaydedilmis kisi.
- `new_anonymous`: Ilk kez gorulen ve yeni face ID verilen anonim yuz.
- `anonymous`: Daha once gorulen anonim face ID ile yeniden eslesen yuz.

`name` ve `metadata` yalnizca `known` durumunda doludur. `confidence`, mevcut
bir kimlikle eslesme skorudur; ilk kez gorulen anonim yuzde eslesme olmadigi
icin `null` olur. Anonim kimlikler
`anonymous_identities`, yuz ornekleri `anonymous_face_embeddings` tablolarinda
saklanir. Bilinen ve anonim yuzler ayni goruntude birlikte bulunabilir.

`POST /faces/enroll`; yeni bir yuzun ad, soyad ve metadata ile kaydini, mevcut
anonim face ID'nin isimlendirilmesini ve kayitli kimlige yeni yuz ornegi
eklenmesini destekler. Anonim kimlik isimlendirilirken mevcut face ID korunur
ve sonraki tanima sonucu `known` olur.

Yeni anonim yuzlerde tam kaynak fotograf yerine bounding box cevresinden
kirpilmis yuz goruntusu saklanir. Ayni goruntunun gereksiz kopyalari eklenmez
ve her anonim kimlik icin en fazla `ANONYMOUS_FACE_MAX_SAMPLES` (varsayilan 5)
gorsel ornegi tutulur. Kimlik isimlendirildiginde bu ornekler ayni face ID
korunarak kisinin referans fotograflarina donusturulur.

Frontend'deki `Kimlikler` ekrani bilinen ve anonim kayitlari listeler; ad veya
face ID ile arama, durum filtresi, bilinen kimligi guncelleme, anonim kimligi
isimlendirme ve kimlik silme islemlerini destekler. Kimlik silinse bile eski
tanima olaylarindaki face ID denetim izi olarak korunur.

Yuz islemi takip endpoint'i: `GET /processes/{processId}`

Face ID gecmis endpoint'i:
`GET /faces/{faceId}/history?limit=20&offset=0`

`detect`, `compare` ve `identify` isteklerinin her biri icin benzersiz UUID
formatinda bir `process_id` uretilir. Bu deger basarili cevaplarda govdede,
butun cevaplarda `X-Process-ID` basliginda doner; dogrulama ve isleme
hatalarinin govdesinde de korunur. Birden fazla yuz iceren tanima istegindeki
olaylar ayni process ID altinda gruplanir. Process sorgusu islem turunu,
durumu, HTTP kodunu, zamanlari, bulunan yuz sayisini, sonucu ve yuz olaylarini
dondurur. `task_detail` alani islem turunu, islenen yuz sayisini ve her yuzun
`face_id`/`status` bilgisini kalici olarak saklar. Ana kayitlar PostgreSQL'deki
`recognition_processes` tablosunda tutulur. Veritabani loglamasi gecici olarak
basarisiz olursa islem sonucu engellenmez; log `data/logs/recognition-processes.jsonl`
dosyasina yedeklenir ve ayni process ID endpoint'iyle sorgulanabilir.

Face ID gecmis sorgusu, bir yuzun gorundugu islemleri en yeniden eskiye
dogru process ID, zaman damgasi, status ve benzerlik bilgisiyle dondurur.
`limit` ve `offset` parametreleri kalici kayitlar arttiginda sayfalama saglar.
Arayuzde Kimlikler ekranindan ayni gecmise ve her process'in detayina erisilir.

API hata cevaplari butun endpoint'lerde ayni sozlesmeyi kullanir:

```json
{
  "error": {
    "code": "CORRUPT_IMAGE",
    "message": "file: Goruntu dosyasi bozuk veya gecersiz.",
    "details": { "field": "file" }
  },
  "process_id": "istege-ait-uuid-veya-null",
  "timestamp": "2026-08-20T07:00:00Z"
}
```

Sabit `error.code` alani istemcinin hata metnine bagimli kalmadan bos dosya,
bozuk goruntu, desteklenmeyen tur, dogrulama, bulunamayan kaynak, cakisma ve
sunucu hatalarini ayirt etmesini saglar. HTTP status kodlari korunur. Face
endpoint'lerinde process ID govdede ve `X-Process-ID` basliginda doner.
Basarili input/output ve hata semalari Swagger/OpenAPI dokumaninda tanimlidir.

Frontend, dosya secimine ek olarak tarayicinin kamera API'siyle fotograf cekip
ayni tanima akisini kullanabilir. Kamera yalnizca pencere acikken calisir ve
cekilen test fotografi kalici olarak kaydedilmez.

Tamamlanan tanima sonuclari, fotograf tutulmadan `recognition_events`
tablosuna kaydedilir. Istatistik ekranindaki toplam, taninan, taninmayan ve
basari orani bu olaylardan hesaplanir.

Referans fotografi yukleme ve iki fotograf karsilastirma islemleri tek yuz
gerektirir; veritabaninda tanima islemi ise goruntudeki tum yuzleri isler.

Kisi endpoint'leri:

- `POST /api/persons`
- `GET /api/persons`
- `GET /api/persons/{person_id}`
- `PATCH /api/persons/{person_id}`
- `DELETE /api/persons/{person_id}`
- `POST /api/persons/{person_id}/face-images`
- `GET /api/persons/{person_id}/face-images`
- `DELETE /api/persons/{person_id}/face-images/{face_image_id}`

Kaydedilen yuz fotograflari MinIO'daki `face-images` bucket'inda tutulur ve
backend tarafindan `/media` adresi altindan sunulur. Embedding verisi birincil
arama icin Qdrant'ta, guvenli fallback icin PostgreSQL'de `vector(512)` olarak
saklanir.

Bir referans fotografi eklenirken yuz, diger kisilerin kayitlariyla
karsilastirilir. Benzerlik `FACE_DUPLICATE_THRESHOLD` degerine ulasirsa farkli
bir kisi altinda ikinci kez kaydedilmesi engellenir. Ayni kisinin kendi kaydina
farkli referans fotograflari eklenebilir.

Backend baslarken Alembic veritabani migration'larini otomatik uygular.

## LFW Veri Seti

LFW klasoru `datasets/lfw` altinda tutulur ve backend container'ina salt
okunur olarak baglanir. Aktarimda kisi adi ve kalici face ID PostgreSQL'e,
orijinal JPG MinIO'ya, 512 boyutlu yuz vektoru PostgreSQL ve Qdrant'a yazilir.
Her dosyanin sonucu `dataset_import_items` tablosunda izlendigi icin komut
kesilirse kaldigi yerden devam eder ve tamamlanan kayitlari cogaltmaz.

On kisilik pilot aktarim:

```powershell
docker compose exec backend python3 -m app.lfw_import --max-people 10
```

Kalan tum veriyi aktarma veya tamamlanma kontrolu:

```powershell
docker compose exec backend python3 -m app.lfw_import
```

PostgreSQL advisory lock ayni anda ikinci bir LFW aktariminin calismasini
engeller. Modelin yuz bulamadigi fotograflar MinIO'da korunur ve takip
tablosunda `no_face` durumuyla raporlanir.

Ilk calistirmada InsightFace modeli indirilir ve kalici Docker volume'unde
saklanir. Backend NVIDIA GPU'yu Docker uzerinden kullanir.

## Benchmark

Benchmark, uretim PostgreSQL, Qdrant ve MinIO kayitlarina yazmadan LFW verisi
uzerinde yuz tespiti, dogrulama, kapali kume kimlik bulma, bilinmeyen kisi
reddetme, esik secimi ve batch performansini olcer. Veri seti container'a salt
okunur baglanir; JSON, CSV ve HTML raporlari `artifacts/benchmarks` altina yazilir.

Hizli ve tekrarlanabilir varsayilan olcum:

```powershell
docker compose exec backend python3 -m benchmark.run
```

Kucuk bir kontrol kosusu:

```powershell
docker compose exec backend python3 -m benchmark.run --max-pairs-per-class 20 --max-identities 20 --max-unknowns 20 --performance-samples 32
```

Resmi LFW ciftlerinin tamamini kullanan genis olcum:

```powershell
docker compose exec backend python3 -m benchmark.run --full
```

## Video kabul testleri

Video kabul araci MinIO'da bulunan tamamlanmis videolari yeniden analiz eder.
PostgreSQL islemi `READ ONLY` olarak acilir; anonim kimlik, Qdrant vektoru,
MinIO gorseli veya video sonucu olusturulmaz. Beklenen face ID ve gorunme
araliklari manifestte tanimlanir. Kimlik yakalama orani, zamansal IoU, yanlis
bilinen kisi, anonim iz sayisi, iz parcalanmasi ve gercek zaman katsayisi
olculur. JSON, CSV ve HTML raporlari `artifacts/video-acceptance/runs` altina
yazilir.

Son tamamlanmis bes videodan guvenli ve devre disi bir taslak olusturma:

```powershell
docker compose exec backend python3 -m benchmark.video_acceptance_run --create-draft /artifacts/video-acceptance/manifest.json --draft-limit 5
```

Taslakta `expectedFaces` alanlari doldurulup test edilecek kayitlarda `enabled`
degeri `true` yapildiktan sonra kabul kosusu:

```powershell
docker compose exec backend python3 -m benchmark.video_acceptance_run --manifest /artifacts/video-acceptance/manifest.json
```

Gercek video baseline testi, tanima davranisini veya veritabanini degistirmeden
ayni videoyu yeniden analiz eder. JSON/CSV/HTML raporlarina her track icin zaman
araligi, gozlem sayisi, tespit ve eslesme guveni ile kisa/dusuk marjli/parcalanmis
track isaretleri eklenir:

```powershell
docker compose exec backend python3 -m benchmark.video_acceptance_run --manifest /artifacts/video-acceptance/real-video-baseline.json
```

Birden fazla tamamlanmis videodan mevcut davranisi sabitleyen salt-okunur
regresyon snapshot'i olusturulabilir. Her `--process-id` ayri bir test vakasi
olur. Bu snapshot elle dogrulanmis ground truth yerine gecmez; sonraki kod ve
performans degisikliklerinin mevcut sonuclari bozup bozmadigini yakalar:

```powershell
docker compose exec backend python3 -m benchmark.video_regression_snapshot --output /artifacts/video-acceptance/regression-baseline.json --process-id PROCESS_UUID_1 --process-id PROCESS_UUID_2
docker compose exec backend python3 -m benchmark.video_acceptance_run --manifest /artifacts/video-acceptance/regression-baseline.json
```

Tum kalite kapilarini tek seferde calistirip tarihli JSON ve HTML raporu
uretmek icin:

```powershell
python scripts/run_quality_report.py
```

Raporlar `artifacts/quality-reports/<tarih>/` altina yazilir. Video kabul
testi salt okunur calisir; arayuz testleri yalnizca kendi gecici kayitlarini
olusturup temizler.

Manifest bicimi icin `backend/benchmark/video_acceptance.example.json` dosyasi
ornek olarak kullanilabilir. Zamanlar saniye cinsinden yazilir ve her case kendi
limitleriyle varsayilan esikleri gecersiz kilabilir.

## Durdurma

```powershell
docker compose down
```
