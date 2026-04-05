import { useEffect, useRef, useState } from 'react'
import Dropzone from './components/Dropzone.jsx'
import './App.css'

function App() {
  const [previewUrl, setPreviewUrl] = useState('')
  const [fileName, setFileName] = useState('')
  const previewRef = useRef('')

  const backendStatus = {
    ready: false,
    detail: 'Upload interaction is live. FastAPI inference wiring comes next.',
    uses_eda_fusion: false,
    gradcam_available: false,
    checkpoint_loaded: false,
  }

  useEffect(() => {
    return () => {
      if (previewRef.current) {
        URL.revokeObjectURL(previewRef.current)
      }
    }
  }, [])

  const handleFileSelected = (file) => {
    if (!file || !file.type.startsWith('image/')) {
      return
    }

    if (previewRef.current) {
      URL.revokeObjectURL(previewRef.current)
    }

    const nextPreview = URL.createObjectURL(file)
    previewRef.current = nextPreview
    setPreviewUrl(nextPreview)
    setFileName(file.name)
  }

  return (
    <main className="app-shell">
      <div className="ambient ambient-cyan" />
      <div className="ambient ambient-orange" />
      <div className="ambient ambient-grid" />

      <section className="masthead">
        <div>
          <span className="eyebrow">VIPER Forensic Engine</span>
          <h1>Live artifact intelligence for Datathon judges.</h1>
          <p className="lead">
            A premium forensic dashboard for instant visual verdicts, confidence
            scoring, and explainable evidence readouts.
          </p>
        </div>

        <div className="status-cluster">
          <div className={`status-pill ${backendStatus.ready ? 'is-live' : ''}`}>
            <span className="status-dot" />
            {backendStatus.ready ? 'Runtime ready' : 'Runtime offline'}
          </div>
          <div className="status-pill">
            {backendStatus.uses_eda_fusion ? 'EDA fusion online' : 'Image-only path'}
          </div>
          <div className="status-pill">
            {backendStatus.gradcam_available ? 'Grad-CAM++ active' : 'Grad-CAM++ unavailable'}
          </div>
        </div>
      </section>

      <section className="command-deck">
        <div className="upload-column glass-panel">
          <div className="panel-header">
            <span className="panel-kicker">Ingress</span>
            <p>{backendStatus.detail}</p>
          </div>

          <Dropzone
            previewUrl={previewUrl}
            fileName={fileName}
            isAnalyzing={false}
            onFileSelected={handleFileSelected}
          />

          <div className="telemetry-strip">
            <div>
              <span>Selection</span>
              <strong>{fileName ? 'Captured' : 'Awaiting'}</strong>
            </div>
            <div>
              <span>Preview</span>
              <strong>{previewUrl ? 'Ready' : 'Idle'}</strong>
            </div>
            <div>
              <span>Pipeline</span>
              <strong>Frontend only</strong>
            </div>
          </div>
        </div>

        <section className="report-card glass-panel">
          <div className="placeholder-panel">
            <span className="panel-kicker">Forensic Report Card</span>
            <h2>{fileName ? 'Image captured for analysis.' : 'Evidence panel framework is in place.'}</h2>
            <p>
              {fileName
                ? 'The next commit will connect this selected image to the backend and render the live forensic response here.'
                : 'The next feature commit will replace this placeholder with the live upload and analysis interface.'}
            </p>
            <div className="placeholder-lines" aria-hidden="true">
              <span />
              <span />
              <span />
            </div>
          </div>
        </section>
      </section>
    </main>
  )
}

export default App
