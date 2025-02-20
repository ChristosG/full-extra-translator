import React from 'react';
import './SummaryModal.css';

interface SummaryModalProps {
  isVisible: boolean;
  isDarkMode: boolean;
  summary: string;
  isLoading: boolean;   // <--- new prop to indicate loading state
  onClose: () => void;
}

const SummaryModal: React.FC<SummaryModalProps> = ({
  isVisible,
  isDarkMode,
  summary,
  isLoading,
  onClose
}) => {
  if (!isVisible) return null;

  return (
    <div className="summary-overlay" onClick={onClose}>
      <div
        className={`summary-modal ${isDarkMode ? 'dark-modal' : 'light-modal'}`}
        onClick={e => e.stopPropagation()}
      >
        <h2 className="summary-title">Bulletpoint Summary</h2>
        <div className="summary-content">
          {isLoading ? (
            <div className="spinner-container">
              <div className="loader"></div>
              <p>Loading summary...</p>
            </div>
          ) : (
            summary
              .split('\n')
              .map((line, idx) => (
                <p key={idx} style={{ margin: '5px 0' }}>
                  {line}
                </p>
              ))
          )}
        </div>
        {!isLoading && (
          <button onClick={onClose} className="summary-close-btn">
            Close
          </button>
        )}
      </div>
    </div>
  );
};

export default SummaryModal;
