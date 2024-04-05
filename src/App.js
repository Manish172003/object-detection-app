import React, { useState } from "react";
import "./App.css";
import * as tf from "@tensorflow/tfjs";
import * as cocoSsd from "@tensorflow-models/coco-ssd";

function App() {
  const [objects, setObjects] = useState([]);

  const handleFileChange = async (e) => {
    const file = e.target.files[0];
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = async () => {
      const image = new Image();
      image.src = reader.result;
      image.onload = async () => {
        await tf.ready(); // Ensure TensorFlow is ready
        const model = await cocoSsd.load();
        const predictions = await model.detect(image);
        setObjects(predictions);
      };
    };
  };

  return (
    <div className="App">
      <h1>Object Detection</h1>
      <input type="file" accept="image/*" onChange={handleFileChange} />
      <div>
        {objects.map((object, index) => (
          <div key={index}>
            <p>
              {object.class} - Confidence: {Math.round(object.score * 100)}%
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;
