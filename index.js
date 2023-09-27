const express = require("express");
const faceapi = require("face-api.js");
require("dotenv").config();
const mongoose = require("mongoose");
const { Canvas, Image } = require("canvas");
const canvas = require("canvas");
const fileUpload = require("express-fileupload");
faceapi.env.monkeyPatch({ Canvas, Image });
var timeout = require('connect-timeout')

const app = express();

app.use(
  fileUpload({
    useTempFiles: true,
  })
);
app.use(timeout('1000s'))
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

      // Detect all faces in the image
      const detections = await faceapi
        .detectAllFaces(img)
        .withFaceLandmarks()
        .withFaceDescriptors();

      if (detections.length > 0) {
        // Get the first detected face landmarks and face descriptor
        const faceLandmarks = detections[0].landmarks;
        const faceDescriptor = detections[0].descriptor;

        // Calculate the center of the left and right eyes
        const leftEye = faceLandmarks.getLeftEye();
        const rightEye = faceLandmarks.getRightEye();
        const eyesCenter = {
          x: (leftEye.x + rightEye.x) / 2,
          y: (leftEye.y + rightEye.y) / 2,
        };

        // Calculate the angle of rotation based on the slope of the line connecting the eyes
        const dx = rightEye.x - leftEye.x;
        const dy = rightEye.y - leftEye.y;
        const angle = Math.atan2(dy, dx);

        // Use the canvas library to perform the rotation and alignment
        const alignedFaceCanvas = canvas.createCanvas(img.width, img.height);
        const context = alignedFaceCanvas.getContext("2d");
        context.translate(eyesCenter.x, eyesCenter.y);
        context.rotate(angle);
        context.drawImage(img, -eyesCenter.x, -eyesCenter.y);

        // Crop the aligned face to a square region around the center
        const cropSize = Math.min(128, Math.min(img.width, img.height));
        const x = eyesCenter.x - cropSize / 2;
        const y = eyesCenter.y - cropSize / 2;
        const alignedFaceImage = canvas.createCanvas(128, 128);
        const alignedContext = alignedFaceImage.getContext("2d");
        alignedContext.drawImage(
          alignedFaceCanvas,
          x < 0 ? -x : 0,
          y < 0 ? -y : 0,
          cropSize,
          cropSize,
          0,
          0,
          128,
          128
        );

        // Add the resized aligned face descriptor to the descriptions array
        descriptions.push(faceDescriptor);
      } else {
        console.log("Terdapat model wajah yang tidak terdeteksi.");
        throw new Error("Terdapat model wajah yang tidak terdeteksi.");
      }
    }

    // Menyimpan data wajah di MongoDB
    const createFace = new FaceModel({
      label: label,
      descriptions: descriptions,
    });
    await createFace.save();
    return true;
  } catch (error) {
    console.log(error);
    throw error;
  }
}

// ... (rest of the code remains unchanged)

/*
Fungsi getDescriptorsFromDB() berguna untuk mengecek wajah apakah sudah dikenali oleh sistem
*/

async function getDescriptorsFromDB(image) {
  try {
    // Get all face data from MongoDB and loop through each to read the data
    let faces = await FaceModel.find();
    for (i = 0; i < faces.length; i++) {
      // Convert face descriptors from Objects to Float32Array
      for (j = 0; j < faces[i].descriptions.length; j++) {
        faces[i].descriptions[j] = new Float32Array(Object.values(faces[i].descriptions[j]));
      }
      faces[i] = new faceapi.LabeledFaceDescriptors(faces[i].label, faces[i].descriptions);
    }

    // Create face matcher to find matching faces with a certain threshold
    const faceMatcher = new faceapi.FaceMatcher(faces, 0.6);

    // Load the image using canvas
    const img = await canvas.loadImage(image);
    let temp = faceapi.createCanvasFromMedia(img);

    // Detect all faces in the image and get their descriptors
    const detections = await faceapi.detectAllFaces(img).withFaceLandmarks().withFaceDescriptors();

    // Crop and resize the face to 128x128
    const faceImages = detections.map((d) => {
      const faceCanvas = faceapi.createCanvasFromMedia(img);
      const ctx = faceCanvas.getContext("2d");

      const x = Math.floor(d.detection.box.x);
      const y = Math.floor(d.detection.box.y);
      const width = Math.floor(d.detection.box.width);
      const height = Math.floor(d.detection.box.height);

      faceCanvas.width = 128;
      faceCanvas.height = 128;

      const scaleX = 128 / width;
      const scaleY = 128 / height;

      ctx.drawImage(
        img,
        x,
        y,
        width,
        height,
        0,
        0,
        Math.floor(width * scaleX),
        Math.floor(height * scaleY)
      );

      return faceCanvas.toDataURL("image/jpeg", 1.0);
    });

    // Mencari wajah yang cocok
    const resizedDetections = faceapi.resizeResults(detections, temp);
    const results = resizedDetections.map((d, i) => ({
      ...faceMatcher.findBestMatch(d.descriptor),
      faceImage: faceImages[i],
    }));

    return results;
  } catch (error) {
    console.log(error);
    throw error;
  }
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

app.post("/recognizing-face", timeout('1000s'), async (req, res) => {
  const File1 = req.files.File1.tempFilePath;
  const File2 = req.files.File2.tempFilePath;
  const File3 = req.files.File3.tempFilePath;
  const File4 = req.files.File4.tempFilePath;
  const File5 = req.files.File5.tempFilePath;
  const label = req.body.label;

  try {
    await uploadLabeledImages([File1, File2, File3, File4, File5], label);
    res.json({ message: "Face data stored successfully" });
  } catch (error) {
    res.status(400).json({ error: "Face not detected in one or more images." });
  }
});

// Fungsi untuk mnengecek apakah dalam sebuah foto terdapat wajah atau tidak?
async function detectFaces(imagePath) {
  const img = await canvas.loadImage(imagePath);
  const detections = await faceapi.detectAllFaces(img);
  return detections.length > 0;
}

function calculateFaceSimilarity(descriptor1, descriptor2) {
  // Menghitung Euclidean distance antara dua deskriptor wajah
  const squaredDistance = faceapi.euclideanDistance(descriptor1, descriptor2);
  const similarity = 1 / (1 + squaredDistance);
  return similarity;
}

/*
Endpoint untuk mengecek wajah dengan sistem face recognition apakah sudah terdaftar pada database
*/

app.post("/recognizer-face", timeout('1000s'), async (req, res) => {
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
      // Calculate similarity and distance
      const similarityResult = filteredResult.map((entry) => ({
        label: entry._label,
        similarity: calculateFaceSimilarity(entry._distance, filteredResult[0]._distance),
        distance: entry._distance,
      }));

      return res.json({ result: similarityResult });
    } else {
      return res.status(404).json({
        message: `Tidak terdapat nama '${label}' pada sistem pengenalan database.`,
      });
    }
  } else {
    return res.json({
      message: "Terdapat error dalam sistem pengenalan wajah.",
    });
  }
});

/*
Endpoint untuk mencari wajah yang sudah terdaftar berdasarkan nama dari pengguna
*/
app.get("/search-face/:label", async (req, res) => {
  const label = req.params.label;

  try {
    // Cari data wajah berdasarkan label yang diberikan dengan proyeksi untuk hanya mengambil field "label"
    const faceData = await FaceModel.findOne({ label }, { label: 1, _id: 0 });

    if (faceData) {
      // Jika ditemukan, kirimkan respons JSON dengan status 200, is_found: true, dan pesan "Data ditemukan dalam database"
      res.status(200).json({ label: label, is_found: true, message: "Data ditemukan dalam database" });
    } else {
      // Jika tidak ditemukan, kirimkan respons JSON dengan status 404, is_found: false, dan pesan "Data tidak ditemukan dalam database"
      res.status(404).json({ label: label, is_found: false, message: "Data tidak ditemukan dalam database" });
    }
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Terjadi kesalahan pada server." });
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
