import React from 'react';
import './SummariesPanel.css';

interface SummaryItem {
  speaker_id: number;
  bulletpoints: string;
}

interface SummariesPanelProps {
  isVisible: boolean;
  isDarkMode: boolean;
  summaries: SummaryItem[];
  onClose: () => void;
}

const SummariesPanel: React.FC<SummariesPanelProps> = ({
  isVisible,
  isDarkMode,
  summaries,
  onClose
}) => {
  if (!isVisible) return null; // Hide entirely if not visible

  return (
    <div className="summaries-overlay" onClick={onClose}>
      <div
        className={`summaries-panel ${isDarkMode ? 'dark-panel' : 'light-panel'}`}
        onClick={(e) => e.stopPropagation()}
      >
        <h2 className="summaries-title">Bulletpoint Summaries</h2>
        <div className="summaries-content">
          {summaries.length === 0 ? (
            <p>No summaries yet.</p>
          ) : (
            summaries.map((item, idx) => (
              <div key={idx} className="summary-item">
                <h4>Speaker {item.speaker_id}:</h4>
                {item.bulletpoints.split('\n').map((line, i) => (
                  <p key={i} style={{ marginLeft: '20px' }}>
                    {line}
                  </p>
                ))}
              </div>
            ))
          )}
        </div>
        <button className="close-summaries-btn" onClick={onClose}>
          Close
        </button>
      </div>
    </div>
  );
};

export default SummariesPanel;
