function EmptyState() {
  return (
    <div className="report-state">
      <div className="state-panel">
        <span className="panel-kicker">Report Panel</span>
        <h3>Awaiting image ingest.</h3>
        <p>
          Upload a candidate image to generate the verdict, confidence score,
          and the four core evidence channels used in the live demo.
        </p>
      </div>
    </div>
  )
}

function LoadingState() {
  return (
    <div className="report-state">
      <div className="state-panel">
        <span className="panel-kicker">Analysis Running</span>
        <h3>VIPER is synthesizing a forensic report.</h3>
        <p>
          Computing ConvNeXt logits, EDA feature fusion, and Grad-CAM attention
          traces.
        </p>
        <div className="loading-bars" aria-hidden="true">
          <span />
          <span />
          <span />
        </div>
      </div>
    </div>
  )
}

function ErrorState({ error }) {
  return (
    <div className="report-state">
      <div className="state-panel is-error">
        <span className="panel-kicker">Pipeline Error</span>
        <h3>Inference could not complete.</h3>
        <p>{error}</p>
      </div>
    </div>
  )
}

function ForensicReportCard({ report, error, fileName, isAnalyzing }) {
  if (isAnalyzing) {
    return (
      <section className="report-card glass-panel">
        <LoadingState />
      </section>
    )
  }

  if (error) {
    return (
      <section className="report-card glass-panel">
        <ErrorState error={error} />
      </section>
    )
  }

  if (!report) {
    return (
      <section className="report-card glass-panel">
        <EmptyState />
      </section>
    )
  }

  const predictionClass = report.predicted_index === 1 ? 'is-ai' : 'is-real'

  return (
    <section className="report-card glass-panel">
      <header className="report-header">
        <div>
          <span className="panel-kicker">Forensic Report Card</span>
          <h2 className="report-title">{report.prediction}</h2>
        </div>

        <div className={`prediction-chip ${predictionClass}`}>
          {report.prediction}
        </div>
      </header>

      <div className="confidence-block">
        <span className="meta-label">Confidence</span>
        <p className="confidence-value">
          <span>{report.confidence_pct}</span>
        </p>
        <p className="confidence-subtext">{report.verdict}</p>
      </div>

      <section className="report-meta">
        <div className="meta-card">
          <span className="meta-label">Uploaded file</span>
          <strong>{fileName || report.filename}</strong>
        </div>
        <div className="meta-card">
          <span className="meta-label">AI probability</span>
          <strong>{(report.ai_probability * 100).toFixed(1)}%</strong>
        </div>
        <div className="meta-card">
          <span className="meta-label">Inference path</span>
          <strong>{report.uses_eda_fusion ? 'Hybrid fusion' : 'ConvNeXt only'}</strong>
        </div>
      </section>

      <section className="evidence-panel">
        <h3>Key Evidence</h3>
        <ul className="evidence-list">
          {report.evidence_breakdown.map((item) => (
            <li
              key={item.id}
              className={`evidence-row is-${item.status}`}
            >
              <div className="evidence-copy">
                <div className="evidence-heading">
                  <span className="evidence-bullet" aria-hidden="true" />
                  <span>{item.label}</span>
                </div>
                <p>{item.detail}</p>
              </div>

              <div className="evidence-metrics">
                <span className="evidence-value">{item.value}</span>
                <div className="evidence-meter" aria-hidden="true">
                  <span style={{ width: `${Math.max(item.score * 100, 8)}%` }} />
                </div>
              </div>
            </li>
          ))}
        </ul>
      </section>

      <footer className="report-footer">
        <p>{report.gradcam_available ? 'Grad-CAM evidence included.' : 'Grad-CAM evidence unavailable in this environment.'}</p>
        <small>{report.model_name} checkpoint loaded for live web inference.</small>
      </footer>
    </section>
  )
}

export default ForensicReportCard
