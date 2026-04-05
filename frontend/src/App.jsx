import './App.css'

function App() {
  const backendStatus = {
    ready: false,
    detail: 'FastAPI integration will be connected in the next frontend step.',
    uses_eda_fusion: false,
    gradcam_available: false,
    checkpoint_loaded: false,
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

          <div className="placeholder-panel">
            <span className="panel-kicker">Upload Surface</span>
            <h2>Dropzone interaction comes next.</h2>
            <p>
              This panel is reserved for image preview, drag-and-drop upload,
              and upload telemetry.
            </p>
            <div className="placeholder-lines" aria-hidden="true">
              <span />
              <span />
              <span />
            </div>
          </div>

          <div className="telemetry-strip">
            <div>
              <span>Endpoint</span>
              <strong>Pending</strong>
            </div>
            <div>
              <span>Checkpoint</span>
              <strong>Pending</strong>
            </div>
            <div>
              <span>Pipeline</span>
              <strong>Pending</strong>
            </div>
          </div>
        </div>

        <section className="report-card glass-panel">
          <div className="placeholder-panel">
            <span className="panel-kicker">Forensic Report Card</span>
            <h2>Evidence panel framework is in place.</h2>
            <p>
              The next feature commit will replace this placeholder with the
              live upload and analysis interface.
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
