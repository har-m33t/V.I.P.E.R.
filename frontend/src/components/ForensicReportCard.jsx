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

function QueuedState({ fileName }) {
  return (
    <div className="report-state">
      <div className="state-panel">
        <span className="panel-kicker">Ready For Analysis</span>
        <h3>{fileName || 'Image staged in the dashboard.'}</h3>
        <p>
          The backend inference hook is the next step. This panel is ready to
          render the live forensic response as soon as the API is connected.
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
          Computing logits, evidence scoring, and the interpretability panel.
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

function ForensicReportCard({
  report,
  error,
  fileName,
  isAnalyzing,
  isQueued,
}) {
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
        {isQueued ? <QueuedState fileName={fileName} /> : <EmptyState />}
      </section>
    )
  }

  return (
    <section className="report-card glass-panel">
      <header className="report-header">
        <div>
          <span className="panel-kicker">Forensic Report Card</span>
          <h2 className="report-title">{report.prediction}</h2>
        </div>

        <div className="prediction-chip">
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
    </section>
  )
}

export default ForensicReportCard
