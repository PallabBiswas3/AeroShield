// frontend/src/App.jsx
import React, { useEffect, useState, useCallback, useRef } from 'react';
import { MapContainer, TileLayer, CircleMarker, Popup, Polyline, useMap } from 'react-leaflet';
import L from 'leaflet';
import axios from 'axios';
import 'leaflet/dist/leaflet.css';

delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
});

const DELHI_STATIONS = [
  { id: 17, name: "R K Puram", lat: 28.5638, lon: 77.1869 },
  { id: 5404, name: "Pusa", lat: 28.6397, lon: 77.1462 },
  { id: 6358, name: "Mandir Marg", lat: 28.6364, lon: 77.2011 },
  { id: 8118, name: "New Delhi", lat: 28.6252, lon: 77.2183 },
  { id: 8119, name: "Anand Vihar", lat: 28.6476, lon: 77.3158 },
  { id: 8120, name: "Punjabi Bagh", lat: 28.6683, lon: 77.1167 },
  { id: 145, name: "Jahangirpuri", lat: 28.7328, lon: 77.1706 },
  { id: 1437, name: "Rohini", lat: 28.7412, lon: 77.1199 },
  { id: 1439, name: "Narela", lat: 28.8527, lon: 77.0911 },
  { id: 1440, name: "Bawana", lat: 28.7913, lon: 77.0425 },
  { id: 1443, name: "Okhla Ph-2", lat: 28.5485, lon: 77.2736 },
  { id: 1444, name: "Wazirpur", lat: 28.7041, lon: 77.1726 },
];

const EMISSION_SOURCES = [
  { id: 1, name: "Bawana Industrial Estate", lat: 28.7720, lon: 77.0398, type: "Heavy Industrial" },
  { id: 2, name: "Narela Industrial Estate", lat: 28.8540, lon: 77.0940, type: "Heavy Industrial" },
  { id: 3, name: "Wazirpur Industrial Area", lat: 28.7060, lon: 77.1700, type: "Industrial Stack" },
  { id: 4, name: "Okhla Industrial Area", lat: 28.5355, lon: 77.2700, type: "Industrial Stack" },
  { id: 5, name: "Jhilmil Industrial Area", lat: 28.6700, lon: 77.3100, type: "Industrial Stack" },
  { id: 6, name: "Mayapuri Industrial Area", lat: 28.6335, lon: 77.1098, type: "Industrial Stack" },
  { id: 7, name: "Delhi Metro Phase IV (Sites)", lat: 28.6800, lon: 77.2200, type: "Construction" },
  { id: 8, name: "Pragati Maidan Redevelopment", lat: 28.6192, lon: 77.2414, type: "Construction" },
];

function WindArrow({ wind }) {
  const map = useMap();
  const markerRef = useRef(null);

  useEffect(() => {
    if (!wind) return;
    if (markerRef.current) {
      map.removeLayer(markerRef.current);
      markerRef.current = null;
    }
    const arrowIcon = L.divIcon({
      className: '',
      html: `<div style="transform: rotate(${wind.direction_deg}deg); font-size: 28px; color: #00ffcc; text-shadow: 0 0 10px #00ffcc99; line-height: 1; user-select: none;">↑</div>`,
      iconSize: [36, 36],
      iconAnchor: [18, 18],
    });
    const centre = map.getCenter();
    const marker = L.marker([centre.lat - 0.08, centre.lng - 0.12], { icon: arrowIcon, interactive: false, zIndexOffset: -1 }).addTo(map);
    markerRef.current = marker;
    return () => { if (markerRef.current) { map.removeLayer(markerRef.current); markerRef.current = null; } };
  }, [map, wind]);
  return null;
}

const getPollutionColor = (pm25) => {
  if (pm25 < 30) return '#00e676';
  if (pm25 < 60) return '#b2ff59';
  if (pm25 < 90) return '#ffee58';
  if (pm25 < 120) return '#ff9800';
  if (pm25 < 250) return '#f44336';
  return '#b71c1c';
};

function plumeTraceLine(hotspotLat, hotspotLon, windDirDeg, lengthDeg = 0.12) {
  const rad = (windDirDeg * Math.PI) / 180;
  return [[hotspotLat, hotspotLon], [hotspotLat + lengthDeg * Math.cos(rad), hotspotLon + lengthDeg * Math.sin(rad)]];
}

const DISPATCH_STATES = { IDLE: 'idle', SENT: 'sent', CONFIRMED: 'confirmed' };
const hourLabel = (h) => `${h === 0 ? 12 : h > 12 ? h - 12 : h}:00 ${h < 12 ? 'AM' : 'PM'}`;
const windCompass = (deg) => ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'][Math.round(deg / 45) % 8];
const priorityColor = (p) => ({ CRITICAL: '#7f0000', HIGH: '#bf360c', MEDIUM: '#1b5e20' }[p] || '#333');

function formatDate(d) {
  return d.toISOString().split('T')[0];
}

export default function App() {
  const [gridData, setGridData] = useState([]);
  const [windData, setWindData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [agentError, setAgentError] = useState(null);
  const [selectedCell, setSelectedCell] = useState(null);
  const [mandateData, setMandateData] = useState(null);
  const [plumeLine, setPlumeLine] = useState(null);
  const [dispatchState, setDispatchState] = useState(DISPATCH_STATES.IDLE);
  const [forecastHour, setForecastHour] = useState(new Date().getHours());
  const [sliderHour, setSliderHour] = useState(new Date().getHours());
  const [showSources, setShowSources] = useState(true);
  const [showStations, setShowStations] = useState(true);

  // ── Feature #2: Model meta ───────────────────────────────────────────────
  const [modelMeta, setModelMeta] = useState(null);

  // ── Feature #3: Date picker ──────────────────────────────────────────────
  const [forecastDate, setForecastDate] = useState(formatDate(new Date()));

  // ── Feature #1: Case log ─────────────────────────────────────────────────
  const [cases, setCases] = useState([]);
  const [showCaseLog, setShowCaseLog] = useState(false);

  const handleSliderChange = (e) => setSliderHour(parseInt(e.target.value));
  const handleSliderRelease = (e) => {
    const h = parseInt(e.target.value);
    setForecastHour(h);
    setSliderHour(h);
    fetchGridData(h, forecastDate);
  };

  const fetchGridData = useCallback(async (hour, dateStr) => {
    const h = hour ?? forecastHour;
    const d = dateStr ?? forecastDate;
    const dateObj = new Date(d + 'T00:00:00');
    const dow = dateObj.getDay();
    const month = dateObj.getMonth() + 1;
    try {
      setLoading(true); setMandateData(null); setSelectedCell(null); setPlumeLine(null); setAgentError(null); setDispatchState(DISPATCH_STATES.IDLE);
      const res = await axios.get('/api/city-grid', { params: { hour: h, day_of_week: dow, month: month } });
      if (res.data.status === 'success') { setGridData(res.data.grid); setWindData(res.data.wind_meta); }
    } catch (err) { alert("Backend offline.\nRun: cd backend && python main.py"); }
    finally { setLoading(false); }
  }, [forecastHour, forecastDate]);

  // Load model info on mount
  useEffect(() => {
    axios.get('/api/model-info')
      .then(r => { if (r.data.status === 'success') setModelMeta(r.data.meta); })
      .catch(() => { });
    fetchGridData(new Date().getHours(), formatDate(new Date()));
  }, []);

  const handleHotspotClick = async (cell) => {
    if (cell.predicted_pm25 < 90) return;
    setSelectedCell(cell); setAnalyzing(true); setMandateData(null); setAgentError(null); setDispatchState(DISPATCH_STATES.IDLE);
    const windDir = windData?.direction_deg ?? cell.wind_direction ?? 315;
    setPlumeLine(plumeTraceLine(cell.lat, cell.lon, windDir));

    const dateObj = new Date(forecastDate + 'T00:00:00');
    const dow = dateObj.getDay();
    const month = dateObj.getMonth() + 1;

    try {
      const res = await axios.post('/api/analyze-hotspot', {
        cell_id: cell.cell_id,
        lat: cell.lat,
        lon: cell.lon,
        wind_speed: windData?.speed_ms ?? cell.wind_speed ?? 4.0,
        wind_direction: windData?.direction_deg ?? cell.wind_direction ?? 315.0,
        hour: forecastHour,
        day_of_week: dow,
        month: month,
      });
      setMandateData(res.data);
    } catch (err) { setAgentError("Agent unreachable. Check backend logs."); }
    finally { setAnalyzing(false); }
  };

  const handleDispatch = async () => {
    if (dispatchState !== DISPATCH_STATES.IDLE) return;
    setDispatchState(DISPATCH_STATES.SENT);
    try {
      const payload = {
        cell_id: mandateData.cell_id,
        lat: selectedCell?.lat,
        lon: selectedCell?.lon,
        predicted_aqi: mandateData.predicted_aqi,
        aqi_label: mandateData.aqi_category?.label,
        primary_violator: mandateData.attribution_matrix[0]?.name,
        confidence_score: mandateData.attribution_matrix[0]?.confidence_score,
        escalation_level: mandateData.automated_mandate?.escalation_level,
        dispatch_priority: mandateData.automated_mandate?.dispatch_priority,
        statute_violated: mandateData.automated_mandate?.statute_violated,
        legal_notice_draft: mandateData.automated_mandate?.legal_notice_draft,
        case_summary: mandateData.automated_mandate?.case_summary,
        attribution_matrix: mandateData.attribution_matrix
      };
      const res = await axios.post('/api/dispatch', payload);
      if (res.data.status === 'success') {
        setDispatchState(DISPATCH_STATES.CONFIRMED);
        // Refresh case log silently
        axios.get('/api/cases', { params: { limit: 50 } })
          .then(r => { if (r.data.status === 'success') setCases(r.data.cases); })
          .catch(() => { });
      }
    } catch (err) {
      setDispatchState(DISPATCH_STATES.IDLE);
      alert('Failed to dispatch case. Check backend logs.');
    }
  };

  const loadCases = () => {
    axios.get('/api/cases', { params: { limit: 50 } })
      .then(r => { if (r.data.status === 'success') setCases(r.data.cases); })
      .catch(() => alert('Failed to load cases.'));
    setShowCaseLog(v => !v);
  };

  const aqiStats = gridData.length > 0 ? {
    severe: gridData.filter(c => c.predicted_pm25 >= 250).length,
    veryPoor: gridData.filter(c => c.predicted_pm25 >= 120 && c.predicted_pm25 < 250).length,
    avgPm25: Math.round(gridData.reduce((s, c) => s + c.predicted_pm25, 0) / gridData.length),
  } : null;

  const modelBadge = modelMeta?.metrics
    ? `Model: RMSE ${modelMeta.metrics.rmse} · MAE ${modelMeta.metrics.mae} · ±15µg/m³ ${modelMeta.metrics.within_15pct}%`
    : null;

  return (
    <div style={{ width: '100vw', height: '100vh', display: 'flex', flexDirection: 'column', backgroundColor: '#0a0a0a', color: 'white', fontFamily: "'Inter','Segoe UI',sans-serif" }}>
      <div style={{ padding: '10px 20px', backgroundColor: '#000', borderBottom: '1px solid #1a1a1a', display: 'flex', justifyContent: 'space-between', alignItems: 'center', zIndex: 1000, gap: '14px', flexWrap: 'wrap' }}>
        <h1 style={{ margin: 0, fontSize: '19px', letterSpacing: '3px', color: '#00ffcc', whiteSpace: 'nowrap', flexShrink: 0 }}>
          AEROSHIELD <span style={{ color: '#444', fontWeight: 300 }}>IQ</span><span style={{ fontSize: '10px', color: '#333', marginLeft: '8px' }}>DELHI v2</span>
        </h1>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flex: 1, maxWidth: '420px' }}>
          <span style={{ fontSize: '11px', color: '#555', whiteSpace: 'nowrap' }}>Date</span>
          <input
            type="date"
            value={forecastDate}
            onChange={(e) => {
              setForecastDate(e.target.value);
              fetchGridData(forecastHour, e.target.value);
            }}
            style={{ backgroundColor: '#111', color: '#00ffcc', border: '1px solid #333', borderRadius: '4px', padding: '3px 6px', fontSize: '12px', cursor: 'pointer' }}
          />
          <span style={{ fontSize: '11px', color: '#555', whiteSpace: 'nowrap', marginLeft: '6px' }}>Forecast</span>
          <input type="range" min={0} max={23} value={sliderHour} onChange={handleSliderChange} onMouseUp={handleSliderRelease} onTouchEnd={handleSliderRelease} style={{ flex: 1, accentColor: '#00ffcc', cursor: 'pointer' }} />
          <span style={{ fontSize: '12px', color: '#00ffcc', whiteSpace: 'nowrap', minWidth: '65px' }}>{hourLabel(sliderHour)}</span>
        </div>
        {windData && (
          <div style={{ display: 'flex', gap: '12px', backgroundColor: '#111', padding: '5px 12px', borderRadius: '5px', border: '1px solid #222', fontSize: '12px', color: '#aaa', whiteSpace: 'nowrap' }}>
            <span>🌬️ <strong style={{ color: '#fff' }}>{windData.speed_ms} m/s</strong></span>
            <span>🧭 <strong style={{ color: '#fff' }}>{windData.direction_deg}° {windCompass(windData.direction_deg)}</strong></span>
            <span style={{ color: '#333', fontSize: '10px' }}>{windData.source}</span>
          </div>
        )}
        {aqiStats && (
          <div style={{ display: 'flex', gap: '6px', fontSize: '11px', whiteSpace: 'nowrap' }}>
            <span style={{ backgroundColor: '#7f0000', padding: '3px 7px', borderRadius: '4px' }}>Severe {aqiStats.severe}</span>
            <span style={{ backgroundColor: '#b71c1c', padding: '3px 7px', borderRadius: '4px' }}>V.Poor {aqiStats.veryPoor}</span>
            <span style={{ backgroundColor: '#333', padding: '3px 7px', borderRadius: '4px' }}>Avg {aqiStats.avgPm25} µg/m³</span>
          </div>
        )}
        <div style={{ display: 'flex', gap: '6px', alignItems: 'center', flexShrink: 0 }}>
          <button onClick={() => setShowStations(v => !v)} style={{ padding: '4px 9px', fontSize: '11px', cursor: 'pointer', borderRadius: '4px', border: '1px solid #333', backgroundColor: showStations ? '#1a3a3a' : '#111', color: showStations ? '#00ffcc' : '#444' }}>Sensors</button>
          <button onClick={() => setShowSources(v => !v)} style={{ padding: '4px 9px', fontSize: '11px', cursor: 'pointer', borderRadius: '4px', border: '1px solid #333', backgroundColor: showSources ? '#3a1a1a' : '#111', color: showSources ? '#ff6060' : '#444' }}>Sources</button>
          <button onClick={loadCases} style={{ padding: '4px 9px', fontSize: '11px', cursor: 'pointer', borderRadius: '4px', border: '1px solid #333', backgroundColor: showCaseLog ? '#1a1a3a' : '#111', color: showCaseLog ? '#66aaff' : '#444' }}>Cases</button>
          <button onClick={() => fetchGridData(forecastHour, forecastDate)} style={{ padding: '6px 12px', fontSize: '12px', fontWeight: 'bold', cursor: 'pointer', backgroundColor: '#00ffcc', color: '#000', border: 'none', borderRadius: '4px' }}>{loading ? '⏳' : '⚡ Forecast'}</button>
        </div>
      </div>
      <div style={{ flex: 1, position: 'relative' }}>
        {loading && (
          <div style={{ position: 'absolute', inset: 0, backgroundColor: 'rgba(0,0,0,0.55)', zIndex: 2000, display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column', gap: '12px' }}>
            <div style={{ color: '#00ffcc', fontSize: '16px', letterSpacing: '2px' }}>⚡ RUNNING NEURAL FORECAST</div>
            <div style={{ color: '#555', fontSize: '12px' }}>LightGBM surrogate · {hourLabel(sliderHour)} · Delhi NCT</div>
          </div>
        )}
        <MapContainer center={[28.6280, 77.2090]} zoom={11} minZoom={10} maxZoom={14} style={{ height: '100%', width: '100%', zIndex: 1 }} zoomControl={false}>
          <TileLayer url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png" attribution="&copy; CARTO" />
          {windData && <WindArrow wind={windData} />}
          {plumeLine && <Polyline positions={plumeLine} pathOptions={{ color: '#ff6600', weight: 2.5, dashArray: '7 5', opacity: 0.9 }} />}
          {showStations && DELHI_STATIONS.map(s => (
            <CircleMarker key={`stn-${s.id}`} center={[s.lat, s.lon]} radius={5} pathOptions={{ fillColor: '#00ccff', color: '#fff', weight: 1.5, fillOpacity: 0.95 }}>
              <Popup><strong>{s.name}</strong><br />CPCB Sensor Node</Popup>
            </CircleMarker>
          ))}
          {showSources && EMISSION_SOURCES.map(src => (
            <CircleMarker key={`src-${src.id}`} center={[src.lat, src.lon]} radius={8} pathOptions={{ fillColor: '#ff3333', color: '#ff0000', weight: 2, fillOpacity: 0.75 }}>
              <Popup><div style={{ color: '#222' }}><strong style={{ color: '#cc0000' }}>⚠ {src.name}</strong><br />Type: {src.type}</div></Popup>
            </CircleMarker>
          ))}
          {gridData.map(cell => (
            <CircleMarker key={`grid-${cell.cell_id}`} center={[cell.lat, cell.lon]} radius={cell.predicted_pm25 >= 120 ? 9 : 7} eventHandlers={{ click: () => handleHotspotClick(cell) }} pathOptions={{ fillColor: getPollutionColor(cell.predicted_pm25), fillOpacity: cell.predicted_pm25 >= 120 ? 0.82 : 0.55, color: getPollutionColor(cell.predicted_pm25), weight: cell.predicted_pm25 >= 90 ? 1.5 : 0 }}>
              <Popup>
                <div style={{ color: '#222', fontSize: '13px' }}>
                  <strong>Grid Cell #{cell.cell_id}</strong><br />PM2.5: <strong>{cell.predicted_pm25.toFixed(1)} µg/m³</strong><br />AQI: <strong>{cell.aqi_category?.label}</strong><br />
                  {cell.predicted_pm25 >= 90 && <em style={{ color: 'red', fontSize: '11px' }}>⚡ Click to dispatch AI agent</em>}
                </div>
              </Popup>
            </CircleMarker>
          ))}
        </MapContainer>

        {/* ── Enforcement Sidebar ─────────────────────────────────────────── */}
        <div style={{ position: 'absolute', top: '14px', right: '14px', width: '365px', backgroundColor: 'rgba(6,6,6,0.97)', padding: '18px', borderRadius: '8px', border: '1px solid #1e1e1e', boxShadow: '0 4px 32px rgba(0,0,0,0.8)', zIndex: 1000, maxHeight: 'calc(100vh - 110px)', overflowY: 'auto' }}>
          <h2 style={{ margin: '0 0 12px 0', fontSize: '14px', color: '#fff', borderBottom: '1px solid #1e1e1e', paddingBottom: '10px', letterSpacing: '1.5px', textTransform: 'uppercase' }}>⚡ Enforcement Agent</h2>
          <div style={{ display: 'flex', gap: '5px', flexWrap: 'wrap', marginBottom: '12px' }}>
            {[['#00e676', 'Good'], ['#b2ff59', 'Satis.'], ['#ffee58', 'Mod.'], ['#ff9800', 'Poor'], ['#f44336', 'V.Poor'], ['#b71c1c', 'Severe']].map(([c, l]) => (
              <span key={l} style={{ display: 'flex', alignItems: 'center', gap: '3px', fontSize: '10px', color: '#666' }}><span style={{ width: '9px', height: '9px', borderRadius: '50%', backgroundColor: c, display: 'inline-block' }} />{l}</span>
            ))}
          </div>
          {!selectedCell && !analyzing && !showCaseLog && (
            <div>
              <p style={{ color: '#444', fontSize: '12px', lineHeight: '1.7', marginBottom: '14px' }}>Drag the hour slider to forecast. Click any <span style={{ color: '#ff9800' }}>orange</span> or <span style={{ color: '#f44336' }}>red</span> grid cell to run reverse-plume attribution and generate a legal mandate.</p>
              <div style={{ borderTop: '1px solid #111', paddingTop: '10px', fontSize: '11px', color: '#333', lineHeight: '2' }}><div>🔵 CPCB sensor stations</div><div>🔴 Industrial / construction sources</div><div>● LightGBM surrogate PM2.5 grid</div></div>
            </div>
          )}
          {analyzing && (
            <div style={{ textAlign: 'center', padding: '20px 0' }}>
              <div style={{ color: '#00ffcc', fontSize: '13px', marginBottom: '8px' }}>⟳ Running Reverse-Plume Math...</div>
              <div style={{ color: '#555', fontSize: '11px', fontStyle: 'italic', marginBottom: '6px' }}>Tracing wind vectors to Delhi industrial zones...</div>
              <div style={{ color: '#666', fontSize: '11px' }}>LangGraph Planner → Legal Drafter → Groq Llama-3</div>
            </div>
          )}
          {agentError && !analyzing && (
            <div style={{ backgroundColor: '#1a0000', border: '1px solid #440000', padding: '12px', borderRadius: '6px', color: '#ff6666', fontSize: '12px', lineHeight: '1.6' }}>
              ⚠ {agentError}<br /><button onClick={() => handleHotspotClick(selectedCell)} style={{ marginTop: '8px', padding: '5px 10px', backgroundColor: '#333', color: '#fff', border: '1px solid #555', borderRadius: '4px', cursor: 'pointer', fontSize: '11px' }}>Retry</button>
            </div>
          )}
          {mandateData && !analyzing && !showCaseLog && (
            <div>
              <div style={{ backgroundColor: '#100000', border: '1px solid #3a0000', padding: '11px', borderRadius: '6px', marginBottom: '12px' }}>
                <div style={{ color: '#ff4444', fontSize: '10px', textTransform: 'uppercase', letterSpacing: '1px', marginBottom: '5px' }}>⚠ Primary Source Identified</div>
                <div style={{ color: '#fff', fontSize: '14px', fontWeight: '600', marginBottom: '3px' }}>{mandateData.attribution_matrix[0]?.name || "Unknown Source"}</div>
                <div style={{ color: '#ff8888', fontSize: '11px' }}>Confidence: <strong>{mandateData.attribution_matrix[0]?.confidence_score || 0}%</strong> · Distance: {mandateData.attribution_matrix[0]?.distance_km || '—'} km</div>
              </div>
              {mandateData.attribution_matrix.length > 1 && (
                <div style={{ marginBottom: '12px' }}>
                  <div style={{ color: '#444', fontSize: '10px', textTransform: 'uppercase', letterSpacing: '1px', marginBottom: '5px' }}>Attribution Ranking</div>
                  {mandateData.attribution_matrix.slice(0, 4).map((src, i) => (
                    <div key={src.source_id} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', padding: '3px 0', borderBottom: '1px solid #111', color: i === 0 ? '#ff8888' : '#444' }}><span>{i + 1}. {src.name}</span><span>{src.confidence_score}%</span></div>
                  ))}
                </div>
              )}
              {mandateData.automated_mandate?.escalation_level && (
                <div style={{ marginBottom: '10px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <div style={{ color: '#444', fontSize: '10px', textTransform: 'uppercase', letterSpacing: '1px' }}>Escalation</div>
                  <span style={{ padding: '3px 9px', borderRadius: '4px', fontSize: '11px', fontWeight: '700', backgroundColor: priorityColor(mandateData.automated_mandate.escalation_level), color: '#fff', letterSpacing: '0.5px' }}>{mandateData.automated_mandate.escalation_level}</span>
                </div>
              )}
              <div style={{ marginBottom: '10px' }}>
                <div style={{ color: '#444', fontSize: '10px', textTransform: 'uppercase', letterSpacing: '1px', marginBottom: '3px' }}>Statute Violated</div>
                <div style={{ color: '#bbb', fontSize: '12px' }}>{mandateData.automated_mandate.statute_violated}</div>
              </div>
              <div style={{ marginBottom: '10px' }}>
                <span style={{ padding: '3px 9px', borderRadius: '4px', fontSize: '11px', fontWeight: '700', backgroundColor: priorityColor(mandateData.automated_mandate.dispatch_priority), color: '#fff' }}>{mandateData.automated_mandate.dispatch_priority} PRIORITY</span>
              </div>
              {mandateData.automated_mandate?.case_summary && (
                <div style={{ marginBottom: '10px', backgroundColor: '#0a0a0a', border: '1px solid #1a1a1a', padding: '8px 10px', borderRadius: '4px' }}>
                  <div style={{ color: '#444', fontSize: '10px', textTransform: 'uppercase', letterSpacing: '1px', marginBottom: '3px' }}>Field Alert Summary</div>
                  <div style={{ color: '#888', fontSize: '11px', lineHeight: '1.5' }}>{mandateData.automated_mandate.case_summary}</div>
                </div>
              )}
              <div style={{ marginBottom: '14px' }}>
                <div style={{ color: '#444', fontSize: '10px', textTransform: 'uppercase', letterSpacing: '1px', marginBottom: '5px' }}>Drafted Legal Notice</div>
                <div style={{ color: '#888', fontSize: '11px', lineHeight: '1.65', fontStyle: 'italic', backgroundColor: '#080808', padding: '10px', borderRadius: '4px', border: '1px solid #161616', whiteSpace: 'pre-wrap', maxHeight: '130px', overflowY: 'auto' }}>{mandateData.automated_mandate.legal_notice_draft}</div>
              </div>
              {dispatchState === DISPATCH_STATES.IDLE && (
                <button onClick={handleDispatch} style={{ width: '100%', padding: '11px', backgroundColor: '#00ffcc', color: '#000', border: 'none', borderRadius: '4px', fontWeight: '700', cursor: 'pointer', fontSize: '13px', letterSpacing: '0.5px' }}>✓ Approve & Dispatch Field Squad</button>
              )}
              {dispatchState === DISPATCH_STATES.SENT && (
                <div style={{ width: '100%', padding: '11px', backgroundColor: '#111', color: '#00ffcc', border: '1px solid #00ffcc44', borderRadius: '4px', textAlign: 'center', fontSize: '12px' }}>⏳ Transmitting to Field Inspector...</div>
              )}
              {dispatchState === DISPATCH_STATES.CONFIRMED && (
                <div style={{ width: '100%', padding: '11px', backgroundColor: '#071407', color: '#00ff66', border: '1px solid #00ff6644', borderRadius: '4px', textAlign: 'center', fontSize: '12px', fontWeight: '700' }}>✅ Squad Dispatched — Case #{mandateData.cell_id}-{String(new Date().getHours()).padStart(2, '0')}{String(new Date().getMinutes()).padStart(2, '0')}</div>
              )}
            </div>
          )}

          {/* ── Case Log Panel ────────────────────────────────────────────── */}
          {showCaseLog && (
            <div>
              <div style={{ color: '#444', fontSize: '10px', textTransform: 'uppercase', letterSpacing: '1px', marginBottom: '8px', borderBottom: '1px solid #1a1a1a', paddingBottom: '6px' }}>
                Dispatched Cases ({cases.length})
              </div>
              {cases.length === 0 && (
                <div style={{ color: '#333', fontSize: '12px', padding: '10px 0' }}>No cases dispatched yet.</div>
              )}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {cases.map((c) => (
                  <div key={c.case_id} style={{ backgroundColor: '#0a0a0a', border: '1px solid #1a1a1a', padding: '10px', borderRadius: '5px', fontSize: '11px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                      <span style={{ color: '#00ffcc', fontWeight: '700' }}>Case #{c.case_id}</span>
                      <span style={{ color: '#444' }}>{new Date(c.created_at).toLocaleString()}</span>
                    </div>
                    <div style={{ color: '#888', marginBottom: '3px' }}>Cell {c.cell_id} · {c.aqi_label} · {c.predicted_aqi?.toFixed?.(1) || c.predicted_aqi} µg/m³</div>
                    <div style={{ color: '#ff8888', marginBottom: '3px' }}>🎯 {c.primary_violator} ({c.confidence_score}%)</div>
                    <div style={{ display: 'flex', gap: '6px' }}>
                      <span style={{ padding: '2px 6px', borderRadius: '3px', fontSize: '10px', fontWeight: '700', backgroundColor: priorityColor(c.escalation_level), color: '#fff' }}>{c.escalation_level}</span>
                      <span style={{ padding: '2px 6px', borderRadius: '3px', fontSize: '10px', fontWeight: '700', backgroundColor: priorityColor(c.dispatch_priority), color: '#fff' }}>{c.dispatch_priority}</span>
                      <span style={{ color: '#333', fontSize: '10px', padding: '2px 0' }}>{c.statute_violated}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* ── Footer ────────────────────────────────────────────────────── */}
        <div style={{ position: 'absolute', bottom: '14px', left: '14px', backgroundColor: 'rgba(0,0,0,0.88)', padding: '7px 12px', borderRadius: '5px', border: '1px solid #1a1a1a', zIndex: 1000, fontSize: '10px', color: '#444' }}>
          AeroShield IQ · Delhi NCT · LightGBM Surrogate · LangGraph + Groq Llama-3
          {aqiStats && (
            <span style={{ marginLeft: '10px' }}>Grid avg: <strong style={{ color: '#aaa' }}>{aqiStats.avgPm25} µg/m³ PM2.5</strong></span>
          )}
          {modelBadge && (
            <span style={{ marginLeft: '10px', color: '#555' }}>· {modelBadge}</span>
          )}
        </div>
      </div>
    </div>
  );
}