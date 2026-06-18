// Top-level UI: webcam capture, onnxruntime-web inference, skeleton + caption overlay.
import { useState } from "react";

// TODO: import { runHolistic } from "./inference/holistic";
// TODO: import { recognize } from "./inference/recognizer"; // onnxruntime-web
// TODO: import { normalizeKeypoints } from "./core_port/normalization";
// TODO: import { ctcCollapse } from "./core_port/ctc_collapse";

export function App() {
  const [caption, setCaption] = useState<string>("");
  const [keypointsOnly, setKeypointsOnly] = useState<boolean>(true);

  // TODO: getUserMedia -> frame loop -> holistic -> normalize -> recognizer -> collapse
  // TODO: render <canvas> skeleton overlay + live caption
  // TODO: A/B viewer (pose vs RGB); gloss timeline; privacy kpts-only toggle

  return (
    <main>
      <h1>sign-rt</h1>
      <p>Real-time sign-language recognition &amp; translation (browser demo)</p>
      <label>
        <input
          type="checkbox"
          checked={keypointsOnly}
          onChange={(e) => setKeypointsOnly(e.target.checked)}
        />
        Keypoints-only (privacy)
      </label>
      <p>Caption: {caption || "(waiting…)"}</p>
    </main>
  );
}
