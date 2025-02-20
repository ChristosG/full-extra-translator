import React, { useState, useEffect, useRef, useCallback } from 'react';
import ReactDOM from 'react-dom/client';
import './App.css';
import DetachedContainer from './DetachedContainer';
import { FaExternalLinkAlt } from 'react-icons/fa';
import SummariesPanel from './SummariesPanel';
import SummaryModal from './SummaryModal';
import { Buffer } from 'buffer';

interface TooltipProps {
  visible: boolean;
  message: string;
  position: React.CSSProperties;
}

interface SummaryData {
  speaker_ids: number[];
  bulletpoints: string;
}


interface TTSData {
  base64_audio: string;
  sample_rate: number;
}

const Tooltip: React.FC<TooltipProps> = ({ visible, message, position }) => {
  if (!visible) return null;

  return (
    <div className="tooltip" style={position}>
      <span className="tooltip-text">{message}</span>
    </div>
  );
};

function App() {
  const [isDarkMode, setIsDarkMode] = useState<boolean>(true);

  const [connectionTooltipVisible, setConnectionTooltipVisible] = useState<boolean>(false);
  const [themeTooltipVisible, setThemeTooltipVisible] = useState<boolean>(false);
  const [betterTranslationTooltipVisible, setBetterTranslationTooltipVisible] = useState<boolean>(false);

  const [currentLineGerman, setCurrentLineGerman] = useState<string>('');
  const [previousLinesGerman, setPreviousLinesGerman] = useState<string[]>([]);

  const [currentLineEnglish, setCurrentLineEnglish] = useState<string>('');
  const [previousLinesEnglish, setPreviousLinesEnglish] = useState<string[]>([]);

  const [useBetterTranslation, setUseBetterTranslation] = useState<boolean>(false);
  const [currentBetterTranslation, setCurrentBetterTranslation] = useState<string>('');
  const [finalizedBetterTranslations, setFinalizedBetterTranslations] = useState<string[]>([]);

  const scrollContainerGermanRef = useRef<HTMLDivElement>(null);
  const scrollContainerEnglishRef = useRef<HTMLDivElement>(null);

  const [isAutoScrollGerman, setIsAutoScrollGerman] = useState<boolean>(true);
  const [isAutoScrollEnglish, setIsAutoScrollEnglish] = useState<boolean>(true);

  const [newMessagesGerman, setNewMessagesGerman] = useState<boolean>(false);
  const [newMessagesEnglish, setNewMessagesEnglish] = useState<boolean>(false);

  const autoScrollTimerGerman = useRef<NodeJS.Timeout | null>(null);
  const autoScrollTimerEnglish = useRef<NodeJS.Timeout | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const retryTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const retryTimeoutDuration = 3000;

  const [detachedGermanWindow, setDetachedGermanWindow] = useState<Window | null>(null);
  const [detachedEnglishWindow, setDetachedEnglishWindow] = useState<Window | null>(null);

  const germanContainerRef = useRef<HTMLDivElement | null>(null);
  const englishContainerRef = useRef<HTMLDivElement | null>(null);
  const germanRootRef = useRef<ReactDOM.Root | null>(null);
  const englishRootRef = useRef<ReactDOM.Root | null>(null);

  const [isSummaryLoading, setIsSummaryLoading] = useState(false);
  const [showSummaryModal, setShowSummaryModal] = useState(false);
  const [summaryResult, setSummaryResult] = useState("");

  const [ttsEnabled, setTtsEnabled] = useState(false);
  const [ttsQueue, setTtsQueue] = useState<TTSData[]>([]);
  const [isPlaying, setIsPlaying] = useState(false);

  const audioCtxRef = useRef<AudioContext | null>(null);
  const currentSourceRef = useRef<AudioBufferSourceNode | null>(null);

  const ttsQueueRef = useRef<TTSData[]>([]);
  const ttsWsRef = useRef<WebSocket | null>(null);

    useEffect(() => {
      ttsQueueRef.current = ttsQueue;
    }, [ttsQueue]);

    useEffect(() => {
      audioCtxRef.current = new AudioContext(); 
    }, []);
    
  const toggleTheme = () => {
    setIsDarkMode((prevMode) => !prevMode);
  };

  const exportTranscription = (lines: string[], filename: string) => {
    const element = document.createElement('a');
    const file = new Blob([lines.join('\n')], { type: 'text/plain' });
    element.href = URL.createObjectURL(file);
    element.download = filename;
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };
  useEffect(() => {
    if (ttsEnabled) {
      const ws = new WebSocket("wss://zelime.duckdns.org/ws-tts");
      ttsWsRef.current = ws;
  
      ws.onopen = () => console.log("[WS-TTS] connected");
      ws.onmessage = (evt) => {
        try {
          const msg = JSON.parse(evt.data);
          if (msg.channel === "tts") {
            const data = msg.data;
            setTtsQueue(prev => [...prev, {
              base64_audio: data.base64_audio,
              sample_rate: data.sample_rate
            }]);
          }
        } catch (err) {
          console.error("[WS-TTS] parse error:", err);
        }
      };
      ws.onclose = () => console.log("[WS-TTS] disconnected");
  
      // Cleanup: close on unmount or if ttsEnabled goes false
      return () => {
        console.log("[WS-TTS] cleaning up...");
        ws.close();
        ttsWsRef.current = null;
      };
    } else {
      // if !ttsEnabled, ensure WS is closed
      if (ttsWsRef.current) {
        ttsWsRef.current.close();
        ttsWsRef.current = null;
      }
      return;
    }
  }, [ttsEnabled]);

  // 2) When queue changes, if ttsEnabled & not playing, call playNextChunk
  useEffect(() => {
    // If TTS is enabled, and we are not currently playing, and we do have something queued:
    if (ttsEnabled && !isPlaying && ttsQueue.length > 0) {
      setIsPlaying(true);
      processTtsQueue();
    }
  }, [ttsEnabled, ttsQueue, isPlaying]);
  
  async function processTtsQueue() {
    // Keep playing until the queue is empty OR TTS is disabled.
    while (ttsEnabled && ttsQueueRef.current.length > 0) {
      const nextChunk = ttsQueueRef.current.shift(); // remove first item
      if (!nextChunk) break;
      await playBase64Float32(nextChunk.base64_audio, nextChunk.sample_rate);
    }
    setIsPlaying(false);
  }
  

  async function playBase64Float32(b64String: string, sampleRate: number) {
    if (!audioCtxRef.current) return;
  
    if (audioCtxRef.current.state === "suspended") {
      await audioCtxRef.current.resume();
    }
  
    const raw = atob(b64String);
    const buf = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i++) {
      buf[i] = raw.charCodeAt(i);
    }
    const float32arr = new Float32Array(buf.buffer);
  
    const audioBuffer = audioCtxRef.current.createBuffer(1, float32arr.length, sampleRate);
    audioBuffer.copyToChannel(float32arr, 0);
  
    return new Promise<void>((resolve) => {
      const source = audioCtxRef.current!.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(audioCtxRef.current!.destination);
  
      currentSourceRef.current = source;
      source.onended = () => {
        currentSourceRef.current = null;
        resolve();
      };
      source.start(0);
    });
  }

  function handleToggleTTS() {
    const newVal = !ttsEnabled;
    if (newVal) {
      // user is enabling TTS
      audioCtxRef.current?.resume().then(() => {
        setTtsEnabled(true);
      });
    } else {
      // user is disabling TTS
      setTtsQueue([]);
      ttsQueueRef.current = [];
      setIsPlaying(false);
      if (currentSourceRef.current) {
        currentSourceRef.current.stop();
        currentSourceRef.current = null;
      }
      setTtsEnabled(false); // triggers the effect to close the WS
    }
  }
  
  
  const initializeWebSocket = () => {
    // const wsProtocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    // const wsPort = '7000';
    // const ws = new WebSocket(`${wsProtocol}://${window.location.hostname}:${wsPort}/ws`);
    
    // const wsProtocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    // const ws = new WebSocket(`${wsProtocol}://${window.location.hostname}/ws`);
    
    const wsProtocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    let wsPort = '';
    
    // Use port 7000 only when running on localhost (development)
    if (window.location.hostname === 'localhost') {
      wsPort = ':7000';
    }
    
    const ws = new WebSocket(`${wsProtocol}://${window.location.hostname}${wsPort}/ws`);
    


    wsRef.current = ws;

    ws.onopen = () => {
      console.log('WebSocket connected');
      setIsConnected(true);
    };

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        const { channel, data } = message; 

        if (channel === 'transcriptions') {
          if (data.transcription) {
            const germanText = data.transcription;
            setCurrentLineGerman(germanText);
            setPreviousLinesGerman((prevLines) => [...prevLines, germanText]);

            if (!isAutoScrollGerman) {
              setNewMessagesGerman(true);
            }
          }
        } else if (channel === 'translations') {
          if (data.translation) {
            const englishText = data.translation;
            setCurrentLineEnglish(englishText);
            setPreviousLinesEnglish((prevLines) => [...prevLines, englishText]);

            if (!isAutoScrollEnglish) {
              setNewMessagesEnglish(true);
            }
          }
        }

         else if (channel === 'better_translations') {
          if (data.translation) {
            const betterText = data.translation;

            if (data.finalized) {
              setFinalizedBetterTranslations((prev) => [...prev, betterText]);
              setCurrentBetterTranslation('');
            } else {
              setCurrentBetterTranslation(betterText);
            }

            if (!isAutoScrollEnglish) {
              setNewMessagesEnglish(true);
            }
          }
        }
      } catch (error) {
        console.error('Error parsing message:', error);
      }
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected, attempting to reconnect...');
      setIsConnected(false);
      scheduleReconnect();
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  };

  const scheduleReconnect = () => {
    if (retryTimeoutRef.current) {
      clearTimeout(retryTimeoutRef.current);
    }
    retryTimeoutRef.current = setTimeout(() => {
      console.log('Reconnecting WebSocket...');
      initializeWebSocket();
    }, retryTimeoutDuration);
  };

  useEffect(() => {
    initializeWebSocket();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (retryTimeoutRef.current) {
        clearTimeout(retryTimeoutRef.current);
      }
      if (detachedGermanWindow) detachedGermanWindow.close();
      if (detachedEnglishWindow) detachedEnglishWindow.close();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Auto-scroll effects for German
  useEffect(() => {
    if (isAutoScrollGerman && scrollContainerGermanRef.current) {
      scrollContainerGermanRef.current.scrollTop = scrollContainerGermanRef.current.scrollHeight;
      setNewMessagesGerman(false);
    }
  }, [previousLinesGerman, isAutoScrollGerman, currentLineGerman]);

  // Auto-scroll effects for English
  useEffect(() => {
    if (isAutoScrollEnglish && scrollContainerEnglishRef.current) {
      scrollContainerEnglishRef.current.scrollTop = scrollContainerEnglishRef.current.scrollHeight;
      setNewMessagesEnglish(false);
    }
  }, [
    previousLinesEnglish,
    finalizedBetterTranslations,
    currentBetterTranslation,
    isAutoScrollEnglish,
    currentLineEnglish,
  ]);

  // Scroll handlers
  const handleScrollGerman = useCallback(
    (event: React.UIEvent<HTMLDivElement, UIEvent>) => {
      const { scrollTop, scrollHeight, clientHeight } = event.currentTarget;
      const isAtBottom = scrollTop + clientHeight >= scrollHeight - 20;

      if (isAtBottom) {
        setIsAutoScrollGerman(true);
        setNewMessagesGerman(false);
      } else {
        setIsAutoScrollGerman(false);
      }

      if (!isAutoScrollGerman) {
        if (autoScrollTimerGerman.current) {
          clearTimeout(autoScrollTimerGerman.current);
        }

        autoScrollTimerGerman.current = setTimeout(() => {
          setIsAutoScrollGerman(true);
        }, 15000);
      }
    },
    [isAutoScrollGerman]
  );

  const handleScrollEnglish = useCallback(
    (event: React.UIEvent<HTMLDivElement, UIEvent>) => {
      const { scrollTop, scrollHeight, clientHeight } = event.currentTarget;
      const isAtBottom = scrollTop + clientHeight >= scrollHeight - 20;

      if (isAtBottom) {
        setIsAutoScrollEnglish(true);
        setNewMessagesEnglish(false);
      } else {
        setIsAutoScrollEnglish(false);
      }

      if (!isAutoScrollEnglish) {
        if (autoScrollTimerEnglish.current) {
          clearTimeout(autoScrollTimerEnglish.current);
        }

        autoScrollTimerEnglish.current = setTimeout(() => {
          setIsAutoScrollEnglish(true);
        }, 15000);
      }
    },
    [isAutoScrollEnglish]
  );

  const [isMobile, setIsMobile] = useState<boolean>(false);

  useEffect(() => {
    const handleResize = () => {
      const width = window.innerWidth;
      setIsMobile(width < 768);
    };

    handleResize();

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, []);

  // Active tab state for mobile
  const [activeTab, setActiveTab] = useState<'german' | 'english'>('german');

  async function handleSummarize() {
    try {
      setIsSummaryLoading(true);
      setShowSummaryModal(true);
      setSummaryResult(""); // Clear old summary if any
  
      const textToSummarize = englishMessages.join("\n"); 
      // or whichever text array you want
  
      const response = await fetch("https://zelime.duckdns.org/summaries", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: textToSummarize })
      });
      const data = await response.json();
  
      setSummaryResult(data.summary || "No summary returned.");
    } catch (err) {
      console.error("Summary request failed:", err);
      setSummaryResult("Error producing summary.");
    } finally {
      setIsSummaryLoading(false);
    }
  }

  // detach transcription windows
  const detachWindow = (
    language: 'german' | 'english',
    messages: string[],
    detachedWindowState: Window | null,
    setDetachedWindowState: React.Dispatch<React.SetStateAction<Window | null>>,
    containerRef: React.RefObject<HTMLDivElement>,
    rootRef: React.MutableRefObject<ReactDOM.Root | null>
  ) => {
    if (detachedWindowState && !detachedWindowState.closed) {
      detachedWindowState.focus();
      return;
    }

    const newWindow = window.open('', '', 'width=600,height=220,left=100,top=300,scrollbars=yes');

    if (newWindow) {
      newWindow.document.title = `Detached ${language === 'german' ? 'German Transcription' : 'English Translation'}`;
      newWindow.document.body.style.margin = '0';
      newWindow.document.body.style.padding = '0';
      newWindow.document.body.style.backgroundColor = isDarkMode ? '#2c2c2c' : '#f9f9f9';
      newWindow.document.body.style.color = isDarkMode ? '#ffffff' : '#333333';

      const container = newWindow.document.createElement('div');
      newWindow.document.body.appendChild(container);
      language === 'german' ? (germanContainerRef.current = container) : (englishContainerRef.current = container);

      const root = ReactDOM.createRoot(container);
      language === 'german' ? (germanRootRef.current = root) : (englishRootRef.current = root);
      root.render(
        <DetachedContainer
          messages={messages}
          isDarkMode={isDarkMode}
          title={language === 'german' ? 'German Transcription' : 'English Translation'}
          language={language}
        />
      );

      setDetachedWindowState(newWindow);

      newWindow.onbeforeunload = () => {
        setDetachedWindowState(null);
        rootRef.current = null;
        language === 'german' ? (germanContainerRef.current = null) : (englishContainerRef.current = null);
      };
    }
  };

  const germanMessages = [...previousLinesGerman, currentLineGerman].filter(Boolean);
  const englishMessages = useBetterTranslation
    ? [...finalizedBetterTranslations, currentBetterTranslation].filter(Boolean)
    : [...previousLinesEnglish, currentLineEnglish].filter(Boolean);

  // update detached windows when messages change
  useEffect(() => {
    if (germanRootRef.current && germanContainerRef.current) {
      if (detachedGermanWindow && !detachedGermanWindow.closed) {
        germanRootRef.current.render(
          <DetachedContainer
            messages={germanMessages}
            isDarkMode={isDarkMode}
            title="German Transcription"
            language="german"
          />
        );
      } else {
        // clean up if the window is closed
        germanRootRef.current.unmount();
        germanRootRef.current = null;
        germanContainerRef.current = null;
        setDetachedGermanWindow(null);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [germanMessages, isDarkMode]);

  useEffect(() => {
    if (englishRootRef.current && englishContainerRef.current) {
      if (detachedEnglishWindow && !detachedEnglishWindow.closed) {
        englishRootRef.current.render(
          <DetachedContainer
            messages={englishMessages}
            isDarkMode={isDarkMode}
            title="English Translation"
            language="english"
          />
        );
      } else {
        // clean up if the window is closed
        englishRootRef.current.unmount();
        englishRootRef.current = null;
        englishContainerRef.current = null;
        setDetachedEnglishWindow(null);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [englishMessages, isDarkMode]);

  return (
    <div className={`container ${isDarkMode ? 'dark-container' : 'light-container'}`}>
      <header className={`header ${isDarkMode ? 'dark-header' : 'light-header'}`}>
        <div className="header-title-container">
          <h1 className={`header-title ${isDarkMode ? 'dark-text-title' : 'light-text-title'}`}>
            test spec
            
          </h1>
          <div
            className="connection-container"
            onMouseEnter={() => setConnectionTooltipVisible(true)}
            onMouseLeave={() => setConnectionTooltipVisible(false)}
            onTouchStart={() => setConnectionTooltipVisible(true)}
            onTouchEnd={() => setConnectionTooltipVisible(false)}
          >
            <div className={`connection-indicator ${isConnected ? 'connected' : 'disconnected'}`} />
            <Tooltip
              visible={connectionTooltipVisible}
              message={isConnected ? 'Connected' : 'Disconnected'}
              position={{ top: '30px', left: '10px' }}
            />
          </div>
          
        </div>
        <button
            style={{
              marginRight: '15px',
              backgroundColor: isDarkMode ? '#333' : '#f0f4f7',
              color: isDarkMode ? '#fff' : '#333',
              border: '1px solid #888',
              padding: '6px 12px',
              borderRadius: '5px',
              cursor: 'pointer'
            }}
            onClick={handleSummarize}
          >
            Summarize
          </button>

          <div>
            <div>
     

              <label>
                <input 
                  type="checkbox"
                  checked={ttsEnabled}
                  onChange={handleToggleTTS}
                />
                Enable TTS
              </label>


     
    </div>
</div>

        <div className="theme-toggle">
          {!isMobile && (
            <span className={`theme-text ${isDarkMode ? 'dark-text' : 'light-text'}`}>
              {isDarkMode ? 'Dark Mode' : 'Light Mode'}
            </span>
          )}
          {isMobile && (
            <div
              className="theme-toggle-mobile"
              onMouseEnter={() => setThemeTooltipVisible(true)}
              onMouseLeave={() => setThemeTooltipVisible(false)}
              onTouchStart={() => setThemeTooltipVisible(true)}
              onTouchEnd={() => setThemeTooltipVisible(false)}
            >
              <label className="switch">
                <input type="checkbox" checked={isDarkMode} onChange={toggleTheme} />
                <span className="slider round"></span>
              </label>
              <Tooltip
                visible={themeTooltipVisible && isMobile}
                message={isDarkMode ? 'Dark Mode' : 'Light Mode'}
                position={{ top: '30px', left: '10px' }}
              />
            </div>
          )}
          {!isMobile && (
            <label className="switch">
              <input type="checkbox" checked={isDarkMode} onChange={toggleTheme} />
              <span className="slider round"></span>
            </label>
          )}
        </div>
      </header>

      {isMobile && (
        <div className="tabs-container">
          <button
            className={`tab-button ${activeTab === 'german' ? 'active-tab-button' : ''}`}
            onClick={() => setActiveTab('german')}
          >
            German
          </button>
          <button
            className={`tab-button ${activeTab === 'english' ? 'active-tab-button' : ''}`}
            onClick={() => setActiveTab('english')}
          >
            English
          </button>
        </div>
      )}

      <main className="main-content">
      <SummaryModal
        isVisible={showSummaryModal}
        isDarkMode={isDarkMode}
        summary={summaryResult}
        isLoading={isSummaryLoading}
        onClose={() => {
          setShowSummaryModal(false);
          setSummaryResult("");
        }}
      />
        {(!isMobile || activeTab === 'german') && (
          <section className={`section ${isDarkMode ? 'dark-section' : 'light-section'}`}>
            <div className="section-header">
              <h2 className={`section-title ${isDarkMode ? 'dark-text-title' : 'light-text-title'}`}>
                German Transcription
              </h2>
              <div className="section-actions">
                <button
                  onClick={() =>
                    exportTranscription(germanMessages, 'German_Transcription.txt')
                  }
                  className="export-button"
                >
                  Export
                </button>
              </div>
            </div>
            <div
              ref={scrollContainerGermanRef}
              className={`history-container ${isDarkMode ? 'dark-scroll-view' : 'light-scroll-view'}`}
              onScroll={handleScrollGerman}
            >
              {previousLinesGerman.length === 0 && !currentLineGerman && (
                <p className={`placeholder-text ${isDarkMode ? 'dark-text' : 'light-text'}`}>
                  No transcriptions yet. Speak to start...
                </p>
              )}
              {previousLinesGerman.map((line, index) => (
                <p key={index} className={`previous-line ${isDarkMode ? 'dark-text' : 'light-text'}`}>
                  {line}
                </p>
              ))}
            </div>
            <div
              className={`current-container ${isDarkMode ? 'dark-current-container' : 'light-current-container'}`}
            >
              <p className={`current-line ${isDarkMode ? 'dark-text' : 'light-text'}`}>
                {currentLineGerman || 'Listening...'}
              </p>
              {/* Detach Button */}
              {!isMobile && (
                <button
                  className="detach-button"
                  onClick={() => {
                    detachWindow(
                      'german',
                      germanMessages,
                      detachedGermanWindow,
                      setDetachedGermanWindow,
                      germanContainerRef,
                      germanRootRef
                    );
                  }}
                  title="Detach German Transcription"
                >
                  <FaExternalLinkAlt />
                </button>
              )}
            </div>
            {newMessagesGerman && isMobile && (
              <button
                className="new-message-button"
                onClick={() => {
                  if (scrollContainerGermanRef.current) {
                    scrollContainerGermanRef.current.scrollTop =
                      scrollContainerGermanRef.current.scrollHeight;
                  }
                  setNewMessagesGerman(false);
                }}
              >
                New Messages
              </button>
            )}
          </section>
        )}

        {(!isMobile || activeTab === 'english') && (
          <section className={`section ${isDarkMode ? 'dark-section' : 'light-section'}`}>
            <div className="section-header">
              <h2 className={`section-title ${isDarkMode ? 'dark-text-title' : 'light-text-title'}`} 
                  style={{display:'flex', flexDirection:'row'}}>
                English Translation

                <div
                  className="better-translation-toggle" style={{marginLeft:'10px'}}
                  onMouseEnter={() => setBetterTranslationTooltipVisible(true)}
                  onMouseLeave={() => setBetterTranslationTooltipVisible(false)}
                  onTouchStart={() => setBetterTranslationTooltipVisible(true)}
                  onTouchEnd={() => setBetterTranslationTooltipVisible(false)}
                >
                  <label className="switch">
                    <input
                      type="checkbox"
                      checked={!useBetterTranslation}
                      onChange={() => setUseBetterTranslation(!useBetterTranslation)}
                    />
                    <span className="slider round"></span>
                  </label>
                  <Tooltip
                    visible={betterTranslationTooltipVisible}
                    message="Just better"
                    position={{ top: '14px', left: '300px' }}
                  />
                </div>
              </h2>
              
              <div className="section-actions">
                <button
                  onClick={() =>
                    exportTranscription(englishMessages, 'English_Translation.txt')
                  }
                  className="export-button"
                >
                  Export
                </button>
                
              </div>
            </div>
            <div
              ref={scrollContainerEnglishRef}
              className={`history-container ${isDarkMode ? 'dark-scroll-view' : 'light-scroll-view'}`}
              onScroll={handleScrollEnglish}
            >
              {useBetterTranslation ? (
                <>
                  {finalizedBetterTranslations.length === 0 && !currentBetterTranslation && (
                    <p className={`placeholder-text ${isDarkMode ? 'dark-text' : 'light-text'}`}>
                      No better translations yet.
                    </p>
                  )}
                  {finalizedBetterTranslations.map((line, index) => (
                    <p key={index} className={`previous-line ${isDarkMode ? 'dark-text' : 'light-text'}`}>
                      {line}
                    </p>
                  ))}
                </>
              ) : (
                <>
                  {previousLinesEnglish.length === 0 && !currentLineEnglish && (
                    <p className={`placeholder-text ${isDarkMode ? 'dark-text' : 'light-text'}`}>
                      No translations yet.
                    </p>
                  )}
                  {previousLinesEnglish.map((line, index) => (
                    <p key={index} className={`previous-line ${isDarkMode ? 'dark-text' : 'light-text'}`}>
                      {line}
                    </p>
                  ))}
                </>
              )}
            </div>
            <div
              className={`current-container ${isDarkMode ? 'dark-current-container' : 'light-current-container'}`}
            >
              <p className={`current-line ${isDarkMode ? 'dark-text' : 'light-text'}`}>
                {useBetterTranslation
                  ? currentBetterTranslation || 'Awaiting better translation...'
                  : currentLineEnglish || 'Awaiting translation...'}
              </p>
              {!isMobile && (
                <button
                  className="detach-button"
                  onClick={() => {
                    detachWindow(
                      'english',
                      englishMessages,
                      detachedEnglishWindow,
                      setDetachedEnglishWindow,
                      englishContainerRef,
                      englishRootRef
                    );
                  }}
                  title="Detach English Translation"
                >
                  <FaExternalLinkAlt />
                </button>
              )}
            </div>
            {newMessagesEnglish && isMobile && (
              <button
                className="new-message-button"
                onClick={() => {
                  if (scrollContainerEnglishRef.current) {
                    scrollContainerEnglishRef.current.scrollTop =
                      scrollContainerEnglishRef.current.scrollHeight;
                  }
                  setNewMessagesEnglish(false);
                }}
              >
                New Messages
              </button>
            )}
          </section>
        )}
      </main>
    </div>
  );
}

export default App;
