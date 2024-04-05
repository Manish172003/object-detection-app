import React, { useState } from "react";
import "./App.css";
import * as tf from "@tensorflow/tfjs";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import * as pdfjs from "pdfjs-dist";
import "pdfjs-dist/build/pdf.worker";

function App() {
  const [personDetected, setPersonDetected] = useState(false);

  const handleFileChange = async (e) => {
    const file = e.target.files[0];
    const reader = new FileReader();
    reader.onload = async () => {
      const pdfData = new Uint8Array(reader.result);
      const pdf = await pdfjs.getDocument({ data: pdfData }).promise;
      const totalPages = pdf.numPages;
      let personDetected = false;

      for (let i = 1; i <= totalPages; i++) {
        const page = await pdf.getPage(i);
        const viewport = page.getViewport({ scale: 1.0 });
        const canvas = document.createElement("canvas");
        const context = canvas.getContext("2d");
        canvas.height = viewport.height;
        canvas.width = viewport.width;

        await page.render({
          canvasContext: context,
          viewport: viewport,
          backgroundTask: false, // Disable background rendering for faster processing
        }).promise;
        const imageData = canvas.toDataURL("image/jpeg");
        const image = new Image();
        image.src = imageData;

        await tf.ready(); // Ensure TensorFlow is ready
        const model = await cocoSsd.load();
        const predictions = await model.detect(image);

        const person = predictions.find(
          (prediction) => prediction.class === "person"
        );
        if (person && person.score > 0.75) {
          personDetected = true;
          break;
        }
      }

      setPersonDetected(personDetected);
    };
    reader.readAsArrayBuffer(file);
  };

  return (
    <div className="App">
      <h1>Person Detection in PDF</h1>
      <input type="file" accept=".pdf" onChange={handleFileChange} />
      {personDetected ? (
        <p>Person detected with confidence greater than 75%!</p>
      ) : (
        <p>No person detected with confidence greater than 75%.</p>
      )}
    </div>
  );
}

export default App;
