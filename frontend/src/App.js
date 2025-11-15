import { useCallback, useEffect, useMemo, useState } from 'react';
import './App.css';

const BACKEND_BASE_URL = 'https://lakers-win-api.onrender.com';

const formatDate = (isoString) => {
  if (!isoString) return '';

  let parsed;
  const dateParts = isoString.split('-').map((part) => Number(part));
  if (dateParts.length === 3 && dateParts.every((n) => !Number.isNaN(n))) {
    const [year, month, day] = dateParts;
    parsed = new Date(year, month - 1, day);
  } else {
    parsed = new Date(isoString);
  }

  if (Number.isNaN(parsed.getTime())) {
    return isoString;
  }

  return parsed.toLocaleDateString(undefined, {
    weekday: 'short',
    month: 'short',
    day: 'numeric',
  });
};
function App() {
  const [nextGame, setNextGame] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [orbKey, setOrbKey] = useState(0);

  const fetchPrediction = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(BACKEND_BASE_URL);
      if (!response.ok) {
        throw new Error(`Backend responded with ${response.status}`);
      }
      const payload = await response.json();
      setNextGame(payload);
      setLastUpdated(new Date());
      setOrbKey(Date.now());
    } catch (err) {
      setError(err.message || 'Unable to reach the backend');
      setNextGame(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchPrediction();
  }, [fetchPrediction]);

  const formattedDate = useMemo(() => formatDate(nextGame?.game_date), [nextGame]);

  const vibeLine = useMemo(() => {
    if (!nextGame) return 'Dialing up the oracle for the next purple & gold storyline.';
    return nextGame.prediction === 1
      ? 'Momentum check: the model smells a W.'
      : 'Warning lights: time to lock in and steal one.';
  }, [nextGame]);

  const predictionOutcome = useMemo(() => {
    if (!nextGame) return null;
    return nextGame.prediction === 1 ? 'Lakers Win' : `${nextGame.opponent} Win`;
  }, [nextGame]);

  return (
    <div className="App">
      <div className="nebula" aria-hidden />
      <div className="nebula accent" aria-hidden />
      <main className="prediction-shell">
        <section className="hero">
          <p className="badge">Chaospanda</p>
          <h1>Lakers Win Predictor</h1>
          <p className="tagline">{vibeLine}</p>
          <div className="hero-actions">
            <button className="primary" onClick={fetchPrediction} disabled={loading}>
              {loading ? 'Summoning data…' : 'Refresh prediction'}
            </button>
            {lastUpdated && (
              <span className="timestamp">Updated {lastUpdated.toLocaleTimeString()}</span>
            )}
          </div>
        </section>

        <section className="prediction-panel">
          <div className="glow-ring" aria-hidden>
            <div className="orb" aria-hidden />
          </div>

          <div className="prediction-card">
            {error && <div className="alert">{error}</div>}

            {nextGame ? (
              <>
                <div key={orbKey} className="probability-orb">
                  <span className="label">model pick</span>
                  <span className="value">{predictionOutcome}</span>
                </div>

                <div className="details-grid compact">
                  <div>
                    <p className="muted">Opponent</p>
                    <h3>{nextGame.opponent}</h3>
                  </div>
                  <div>
                    <p className="muted">Date</p>
                    <h3>{formattedDate}</h3>
                  </div>
                </div>
              </>
            ) : (
              <div className="placeholder">
                {loading ? 'Calibrating the model…' : 'Ping the backend to reveal the next matchup.'}
              </div>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;


