import { useEffect, useRef, useState } from 'react'
import Dropzone from './components/Dropzone.jsx'
import ForensicReportCard from './components/ForensicReportCard.jsx'
import './App.css'

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, '') ||
  'http://127.0.0.1:8000'

function App() {
  const [report, setReport] = useState(null)
  const [previewUrl, setPreviewUrl] = useState('')
  const [fileName, setFileName] = useState('')
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [error, setError] = useState('')
  const [backendStatus, setBackendStatus] = useState({
    ready: false,
    detail: 'Connecting to VIPER runtime...',
    uses_eda_fusion: false,
    gradcam_available: false,
    checkpoint_loaded: false,
  })

  const abortRef = useRef(null)
  const previewRef = useRef('')

  useEffect(() => {
    const controller = new AbortController()

    const checkHealth = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/health`, {
          signal: controller.signal,
        })
        const payload = await response.json()
        if (!response.ok) {
          throw new Error(payload.detail || 'Backend health check failed.')
        }
        setBackendStatus(payload)
      } catch (fetchError) {
        if (fetchError.name === 'AbortError') {
          return
        }
        setBackendStatus({
          ready: false,
          detail: 'Backend unavailable. Start FastAPI on port 8000.',
          uses_eda_fusion: false,
          gradcam_available: false,
          checkpoint_loaded: false,
        })
      }
    }

    checkHealth()
    return () => controller.abort()
  }, [])

  useEffect(() => {
    return () => {
      abortRef.current?.abort()
      if (previewRef.current) {
        URL.revokeObjectURL(previewRef.current)
      }
    }
  }, [])

  const updatePreview = (file) => {
    if (previewRef.current) {
      URL.revokeObjectURL(previewRef.current)
    }
    const nextPreview = URL.createObjectURL(file)
    previewRef.current = nextPreview
    setPreviewUrl(nextPreview)
  }

  const handleFileSelected = async (file) => {
    if (!file) {
      return
    }

    if (!file.type.startsWith('image/')) {
      setError('Only image uploads are supported.')
      return
    }

    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller

    updatePreview(file)
    setFileName(file.name)
    setReport(null)
    setError('')
    setIsAnalyzing(true)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        body: formData,
        signal: controller.signal,
      })
      const payload = await response.json()
      if (!response.ok) {
        throw new Error(payload.detail || 'Prediction request failed.')
      }
      if (abortRef.current === controller) {
        setReport(payload)
        setBackendStatus((current) => ({
          ...current,
          ready: true,
          detail: 'VIPER runtime ready',
        }))
      }
    } catch (requestError) {
      if (requestError.name === 'AbortError') {
        return
      }
      if (abortRef.current === controller) {
        setError(requestError.message)
      }
    } finally {
      if (abortRef.current === controller) {
        abortRef.current = null
        setIsAnalyzing(false)
      }
    }
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
            Upload one image and inspect the model verdict, confidence band, and
            the forensic signals that drove the call.
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
            isAnalyzing={isAnalyzing}
            onFileSelected={handleFileSelected}
          />

          <div className="telemetry-strip">
            <div>
              <span>Endpoint</span>
              <strong>{API_BASE_URL}</strong>
            </div>
            <div>
              <span>Checkpoint</span>
              <strong>{backendStatus.checkpoint_loaded ? 'Loaded' : 'Missing'}</strong>
            </div>
            <div>
              <span>Pipeline</span>
              <strong>{backendStatus.uses_eda_fusion ? 'Hybrid' : 'CNN'}</strong>
            </div>
          </div>
        </div>

        <ForensicReportCard
          report={report}
          error={error}
          fileName={fileName}
          isAnalyzing={isAnalyzing}
        />
      </section>
    </main>
  )
}

export default App
