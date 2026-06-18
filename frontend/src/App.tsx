// Top-level UI: webcam capture, onnxruntime-web inference, skeleton + caption overlay.
import { useCallback, useRef, useState } from "react";
import type { HolisticLandmarker } from "@mediapipe/tasks-vision";

import { createHolistic, packResult } from "./inference/holistic";
import { Recognizer } from "./inference/recognizer";
import { drawSkeleton } from "./inference/skeleton";
import { idsToGloss, translate } from "./inference/gloss";
import { normalizeKeypoints } from "./core_port/normalization";

const INFER_EVERY = 8; // run the recognizer every N frames (~3 Hz at 25 FPS)

export function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const holisticRef = useRef<HolisticLandmarker | null>(null);
  const recognizerRef = useRef<Recognizer | null>(null);
  const rafRef = useRef<number>(0);
  const frameNo = useRef<number>(0);

  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState("idle");
  const [gloss, setGloss] = useState("");
  const [caption, setCaption] = useState("");
  const [keypointsOnly, setKeypointsOnly] = useState(true);

  const loop = useCallback(async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const holistic = holisticRef.current;
    const recognizer = recognizerRef.current;
    if (!video || !canvas || !holistic || !recognizer) return;

    if (video.readyState >= 2) {
      const result = holistic.detectForVideo(video, performance.now());
      const raw = packResult(result);

      const ctx = canvas.getContext("2d");
      if (ctx) drawSkeleton(ctx, raw);

      recognizer.push(normalizeKeypoints(raw));

      if (frameNo.current % INFER_EVERY === 0) {
        const ids = await recognizer.infer();
        const g = idsToGloss(ids);
        setGloss(g);
        setCaption(await translate(g));
      }
      frameNo.current += 1;
    }
    rafRef.current = requestAnimationFrame(() => void loop());
  }, []);

  const start = useCallback(async () => {
    try {
      setStatus("loading models…");
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
      const video = videoRef.current!;
      video.srcObject = stream;
      await video.play();

      holisticRef.current = await createHolistic();
      const recognizer = new Recognizer();
      await recognizer.load();
      recognizerRef.current = recognizer;

      setRunning(true);
      setStatus("running");
      frameNo.current = 0;
      rafRef.current = requestAnimationFrame(() => void loop());
    } catch (err) {
      setStatus(`error: ${(err as Error).message}`);
    }
  }, [loop]);

  const stop = useCallback(() => {
    cancelAnimationFrame(rafRef.current);
    const video = videoRef.current;
    const stream = video?.srcObject as MediaStream | null;
    stream?.getTracks().forEach((t) => t.stop());
    if (video) video.srcObject = null;
    recognizerRef.current?.reset();
    setRunning(false);
    setStatus("stopped");
  }, []);

  return (
    <main style={{ fontFamily: "system-ui, sans-serif", maxWidth: 760, margin: "2rem auto", padding: "0 1rem" }}>
      <h1>sign-rt</h1>
      <p>Real-time sign-language recognition &amp; translation (browser demo)</p>

      <div style={{ display: "flex", gap: "0.75rem", alignItems: "center", marginBottom: "1rem" }}>
        {!running ? (
          <button onClick={() => void start()}>Start camera</button>
        ) : (
          <button onClick={stop}>Stop</button>
        )}
        <label style={{ userSelect: "none" }}>
          <input
            type="checkbox"
            checked={keypointsOnly}
            onChange={(e) => setKeypointsOnly(e.target.checked)}
          />{" "}
          Keypoints-only (privacy)
        </label>
        <span style={{ marginLeft: "auto", color: "#888" }}>{status}</span>
      </div>

      <div style={{ position: "relative", width: 640, height: 480, background: "#111", borderRadius: 8, overflow: "hidden" }}>
        <video
          ref={videoRef}
          width={640}
          height={480}
          playsInline
          muted
          style={{ position: "absolute", inset: 0, visibility: keypointsOnly ? "hidden" : "visible" }}
        />
        <canvas ref={canvasRef} width={640} height={480} style={{ position: "absolute", inset: 0 }} />
      </div>

      <p style={{ marginTop: "1rem" }}>
        <strong>Gloss:</strong> <code>{gloss || "(waiting…)"}</code>
      </p>
      <p>
        <strong>Caption:</strong> {caption || "(waiting…)"}
      </p>
      <p style={{ color: "#aa7", fontSize: "0.85rem" }}>
        Note: the served ONNX recognizer is untrained (random weights) until Phase&nbsp;2 — gloss/caption
        are structurally live but not yet meaningful.
      </p>
    </main>
  );
}
