const express = require("express");
const faceapi = require("face-api.js");
require("dotenv").config();
const mongoose = require("mongoose");
const { Canvas, Image } = require("canvas");
const canvas = require("canvas");
const fileUpload = require("express-fileupload");
faceapi.env.monkeyPatch({ Canvas, Image });

const app = express();

app.use(
  fileUpload({
    useTempFiles: true,
  })
);
/*
  Branch = detectAllFace-recognizer
/*
Load Models
1. Memuat model dari faceRecognitionNet yang bertugas untuk pengenalan dan identifikasi wajah pada gambar yang diberikan
2. Memuat model dari faceLandmark68Net yang mampu mendeteksi 68 landmark atau titik kunci pada wajah seperti hidung, mata, mulut, dan lainnya.
3. Memuat model dari ssdMobilenetv1 yang berfungsi deteksi objek dan juga menemukan berbagai objek, termasuk wajah dalam sebuah gambar
 - faceapi.nets merupakan namespace yang disediakan oleh library 'face-api.js' yang berisikan model-model yang sudah terlatih dari folder "models"
*/
async function LoadModels() {
  await faceapi.nets.faceRecognitionNet.loadFromDisk(__dirname + "/models");
  await faceapi.nets.faceLandmark68Net.loadFromDisk(__dirname + "/models");
  await faceapi.nets.ssdMobilenetv1.loadFromDisk(__dirname + "/models");
}
LoadModels();

/*
Membuat collection baru pada MongoDB
*/
const faceSchema = new mongoose.Schema({
  label: {
    type: String,
    required: true,
    unique: true,
  },
  descriptions: {
    type: Array,
    required: true,
  },
});

const FaceModel = mongoose.model("Face", faceSchema);

/*
Fungsi uploadLabeledImages() berguna untuk mengupload gambar yang akan diregistrasikan
*/
async function uploadLabeledImages(images, label) {
  try {
    let counter = 0;
    const descriptions = [];
    // Melakukan looping berdasarkan berapa banyak jumlah gambar yang diupload
    for (let i = 0; i < images.length; i++) {
      const img = await canvas.loadImage(images[i]);
      counter = (i / images.length) * 100;
      console.log(`Progress = ${counter}%`);
      // Baca setiap wajah dan simpan deskripsi wajah di array deskripsi
      const detections = await faceapi
        .detectSingleFace(img)
        .withFaceLandmarks()
        .withFaceDescriptor();

      if (detections) {
        descriptions.push(detections.descriptor);
      } else {
        console.log("Terdapat model wajah yang tidak terdeteksi.");
        return { error: "Terdapat model wajah yang tidak terdeteksi." };
      }
    }

    // Menyimpan data wajah di MongoDB
    const createFace = new FaceModel({
      label: label,
      descriptions: descriptions,
    });
    await createFace.save();
    return { success: true };
  } catch (error) {
    console.log(error);
    return { error: "Something went wrong while processing the images." };
  }
}

/*
Fungsi getDescriptorsFromDB() berguna untuk mengecek wajah apakah sudah dikenali oleh sistem
*/
async function getDescriptorsFromDB(image) {
  // Dapatkan semua data wajah dari mongodb dan melakukan looping terhadap masing-masing untuk membaca data
  let faces = await FaceModel.find();
  for (i = 0; i < faces.length; i++) {
    // Ubah deskriptor data wajah dari Objects ke tipe Float32Array
    for (j = 0; j < faces[i].descriptions.length; j++) {
      faces[i].descriptions[j] = new Float32Array(
        Object.values(faces[i].descriptions[j])
      );
    }
    faces[i] = new faceapi.LabeledFaceDescriptors(
      faces[i].label,
      faces[i].descriptions
    );
  }

  // Muat pencocokan wajah untuk menemukan wajah yang cocok ditambah berapa banyak batas nilai bobot yang diperkenankan
  const faceMatcher = new faceapi.FaceMatcher(faces, 0.6);

  // Baca gambar menggunakan canvas
  const img = await canvas.loadImage(image);
  let temp = faceapi.createCanvasFromMedia(img);
  // Memproses gambar untuk model yang tersedia
  const displaySize = { width: img.width, height: img.height };
  faceapi.matchDimensions(temp, displaySize);

  // Mencari wajah yang cocok
  const detections = await faceapi
    .detectAllFaces(img)
    .withFaceLandmarks()
    .withFaceDescriptors();
  const resizedDetections = faceapi.resizeResults(detections, displaySize);
  const results = resizedDetections.map((d) =>
    faceMatcher.findBestMatch(d.descriptor)
  );
  return results;
}

// Root Endpoint
app.get("/", (_, res) => {
  res.json({
    message: "Face Recognition with Express JS and MongoDB from Face API",
    status: 200,
  });
});

/*
Endpoint untuk mendaftarkan wajah kedalam database
*/
app.post("/recognizing-face", async (req, res) => {
  const File1 = req.files.File1.tempFilePath;
  const File2 = req.files.File2.tempFilePath;
  const File3 = req.files.File3.tempFilePath;
  const File4 = req.files.File4.tempFilePath;
  const File5 = req.files.File5.tempFilePath;
  const label = req.body.label;
  let result = await uploadLabeledImages(
    [File1, File2, File3, File4, File5],
    label
  );
  if (result) {
    res.json({ message: "Face data stored successfully" });
  } else {
    res.json({ message: "Something went wrong, please try again." });
  }
});

// Fungsi untuk mnengecek apakah dalam sebuah foto terdapat wajah atau tidak?
async function detectFaces(imagePath) {
  const img = await canvas.loadImage(imagePath);
  const detections = await faceapi.detectAllFaces(img);
  return detections.length > 0;
}

/*
Endpoint untuk mengecek wajah dengan sistem face recognition apakah sudah terdaftar pada database
*/

app.post("/recognizer-face", async (req, res) => {
  const { label } = req.body;

  if (label === undefined) {
    return res.status(400).json({ message: "Harap tambahkan parameter nama." });
  }

  const File1 = req.files.File1.tempFilePath;
  const isFace = await detectFaces(File1);

  if (isFace) {
    let result = await getDescriptorsFromDB(File1);

    const filteredResult = result.filter((entry) => entry._label === label);

    if (filteredResult.length > 0) {
      return res.json({ result: filteredResult });
    } else {
      return res.status(404).json({ message: `Tidak terdapat nama '${label}' pada sistem pengenalan database.` });
    }
  } else {
    return res.json({ message: "Terdapat error dalam sistem pengenalan wajah." });
  }
});



/*
Konfigurasi MongoDB
*/
mongoose
  .connect(process.env.MONGO_CONNECTION_STRING, {
    useNewUrlParser: true,
    useUnifiedTopology: true,
    useCreateIndex: true,
  })
  .then(() => {
    app.listen(process.env.PORT || 5000);
    console.log("Server is Running and MongoDB is Connected!");
  })
  .catch((err) => {
    console.log(err);
  });
