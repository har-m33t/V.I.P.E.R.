import { useRef, useState } from 'react'

function Dropzone({ previewUrl, fileName, isAnalyzing, onFileSelected }) {
  const inputRef = useRef(null)
  const [isActive, setIsActive] = useState(false)

  const handleFiles = (fileList) => {
    const [file] = fileList || []
    if (file) {
      onFileSelected(file)
    }
  }

  const openPicker = () => {
    inputRef.current?.click()
  }

  return (
    <div
      className={[
        'dropzone',
        isActive ? 'is-active' : '',
        previewUrl ? 'is-previewing' : '',
      ]
        .filter(Boolean)
        .join(' ')}
      onDragEnter={(event) => {
        event.preventDefault()
        setIsActive(true)
      }}
      onDragOver={(event) => {
        event.preventDefault()
        setIsActive(true)
      }}
      onDragLeave={(event) => {
        event.preventDefault()
        setIsActive(false)
      }}
      onDrop={(event) => {
        event.preventDefault()
        setIsActive(false)
        handleFiles(event.dataTransfer.files)
      }}
    >
      {previewUrl ? (
        <img
          className="dropzone-preview"
          src={previewUrl}
          alt={fileName || 'Uploaded preview'}
        />
      ) : null}
      <div className="dropzone-overlay" />
      <div className="dropzone-scanline" />

      <div className="dropzone-body">
        <div className="dropzone-copy">
          <div className="dropzone-icon" aria-hidden="true">
            <svg width="30" height="30" viewBox="0 0 24 24" fill="none">
              <path
                d="M12 16V6M12 6L8 10M12 6L16 10"
                stroke="currentColor"
                strokeWidth="1.6"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <path
                d="M6 17.5C4.619 17.5 3.5 16.381 3.5 15C3.5 13.774 4.383 12.754 5.548 12.542C5.849 10.008 7.99 8 10.605 8C12.692 8 14.472 9.286 15.21 11.106C15.48 11.036 15.764 11 16.055 11C17.959 11 19.5 12.541 19.5 14.445C19.5 16.349 17.959 17.89 16.055 17.89H9.5"
                stroke="currentColor"
                strokeWidth="1.6"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </div>

          <div>
            <span className="panel-kicker">Upload Evidence</span>
            <h2>
              {isAnalyzing
                ? 'Scanning artifact lattice...'
                : previewUrl
                  ? 'Preview locked. Analysis wiring comes next.'
                  : 'Drag an image into the forensic sandbox.'}
            </h2>
          </div>

          <p>
            VIPER now supports drag-and-drop image intake and live preview in
            the dashboard shell.
          </p>

          <div className="dropzone-meta">
            <span className="meta-chip">PNG / JPG / WEBP</span>
            <span className="meta-chip">Single frame</span>
            <span className="meta-chip">Preview ready</span>
          </div>
        </div>

        <div className="dropzone-actions">
          <button
            className="dropzone-button"
            type="button"
            onClick={openPicker}
          >
            {previewUrl ? 'Choose another image' : 'Select image'}
          </button>

          <p className="dropzone-note">
            {fileName
              ? `Current file: ${fileName}`
              : 'The selected image stays in the local frontend state for now.'}
          </p>
        </div>
      </div>

      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        hidden
        onChange={(event) => {
          handleFiles(event.target.files)
          event.target.value = ''
        }}
      />
    </div>
  )
}

export default Dropzone
