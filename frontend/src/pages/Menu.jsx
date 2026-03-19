import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import apiClient from '../apiClient';

const Menu = () => {
    const [currentTime, setCurrentTime] = useState(new Date());
    const [latency, setLatency] = useState(null);
    const [apiOnline, setApiOnline] = useState(true);
    const [dbOnline, setDbOnline] = useState(true);
    const [wealthData, setWealthData] = useState({
        cumsum_twr: 0,
        daily_twr: 0,
        ratios: [0, 0, 0, 0, 0]
    });
    const [marketPulseData, setMarketPulseData] = useState([]);
    const [assetGrowthData, setAssetGrowthData] = useState({
        growth: 0,
        net_profit: 0,
        sparkline: []
    });

    useEffect(() => {
        const timer = setInterval(() => setCurrentTime(new Date()), 1000);

        // Latency and Health Check
        const checkHealth = async () => {
            const start = performance.now();
            try {
                const response = await apiClient.get('/health');
                const end = performance.now();
                setLatency(Math.round(end - start));
                setApiOnline(true);
                setDbOnline(response.data.db_status === 'ok');
            } catch (err) {
                console.error("Health check failed", err);
                setApiOnline(false);
                setDbOnline(false);
                setLatency(null);
            }
        };

        const fetchWealthData = async () => {
            try {
                const response = await apiClient.get('/wealth_index');
                const data = response.data;
                // data looks like: [{daily_twr: ...}, {ratio0: ...}, ...]
                const mainStats = data[0];
                const ratios = data.slice(1).map(item => Object.values(item)[0]);

                setWealthData({
                    cumsum_twr: mainStats.cumsum_twr,
                    daily_twr: mainStats.daily_twr,
                    ratios: ratios
                });
            } catch (err) {
                console.error("Failed to fetch wealth data", err);
            }
        };

        const fetchMarketPulse = async () => {
            try {
                const response = await apiClient.get('/market_pulse');
                setMarketPulseData(response.data);
            } catch (err) {
                console.error("Failed to fetch market pulse data", err);
            }
        };

        const fetchAssetGrowthData = async () => {
            try {
                const response = await apiClient.get('/asset_growth_index');
                const data = response.data;
                // Process list of dicts: [{'growth': 36.4}, {'net_profit': 6347560.0}, {'net_profit0': 55.0}, ...]
                let growth = 0;
                let net_profit = 0;
                let sparkline = [];

                data.forEach(item => {
                    const key = Object.keys(item)[0];
                    if (key === 'growth') growth = item[key];
                    else if (key === 'net_profit') net_profit = item[key];
                    else if (key.startsWith('net_profit')) sparkline.push(item[key]);
                });

                setAssetGrowthData({
                    growth,
                    net_profit,
                    sparkline
                });
            } catch (err) {
                console.error("Failed to fetch asset growth data", err);
            }
        };

        checkHealth(); // Initial check
        fetchWealthData(); // Initial wealth data fetch
        fetchMarketPulse(); // Initial market pulse fetch
        fetchAssetGrowthData(); // Initial asset growth fetch
        const healthInterval = setInterval(checkHealth, 30000); // Every 30 seconds

        return () => {
            clearInterval(timer);
            clearInterval(healthInterval);
        };
    }, []);

    const formatDate = (date) => {
        return date.toLocaleDateString('ja-JP', {
            year: 'numeric',
            month: 'long',
            day: 'numeric',
            weekday: 'long'
        });
    };

    const formatTime = (date) => {
        return date.toLocaleTimeString('ja-JP', {
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
    };

    return (
        <div className="main menu-dashboard">
            {/* Header Section */}
            <div className="menu-header-card">
                <h1 className="premium-title">PORTFOLIO <span className="nexus-text">NEXUS</span></h1>
                <p className="subtitle">Operational Command & Intelligence</p>
            </div>

            {/* Welcome & Time Card */}
            <div className="bento-card welcome-card">
                <div className="card-glass">
                    <span className="card-tag">STATUS</span>
                    <h4 className="compact-welcome">Welcome Back</h4>
                    <div className="clock-display compact">
                        <div className="time">{formatTime(currentTime)}</div>
                        <div className="date">{formatDate(currentTime)}</div>
                    </div>
                </div>
            </div>

            {/* Asset Growth Index Card (NEW) */}
            <div className="bento-card growth-card">
                <div className="card-glass">
                    <span className="card-tag">GROWTH</span>
                    <h3>Asset Growth Index</h3>
                    <div className="kpi-value">
                        {assetGrowthData.net_profit.toLocaleString()}
                        <span className="unit">JPY</span>
                    </div>
                    <div className={`kpi-trend ${assetGrowthData.growth >= 0 ? 'positive' : 'negative'}`}>
                        {assetGrowthData.growth >= 0 ? '+' : ''}{assetGrowthData.growth}% (YTD)
                    </div>
                    <div className="mini-chart">
                        <svg viewBox="0 0 100 40" preserveAspectRatio="none" style={{width: '100%', height: '100%'}}>
                            <polyline
                                fill="none"
                                stroke="#bef106"
                                strokeWidth="2"
                                points={assetGrowthData.sparkline.map((val, idx) => 
                                    `${(idx / (assetGrowthData.sparkline.length - 1)) * 100},${40 - (val / 100) * 40}`
                                ).join(' ')}
                            />
                        </svg>
                    </div>
                </div>
            </div>

            {/* Net Profit Card (Renamed/Repurposed) */}
            <div className="bento-card kpi-card">
                <div className="card-glass">
                    <span className="card-tag">PERFORMANCE</span>
                    <h3>Net Profit (YTD)</h3>
                    <div className="kpi-value">
                        {wealthData.cumsum_twr >= 0 ? '+' : ''}
                        {(wealthData.cumsum_twr * 100).toFixed(1)}
                        <span className="unit">%</span>
                    </div>
                    <div className={`kpi-trend ${wealthData.daily_twr >= 0 ? 'positive' : 'negative'}`}>
                        {wealthData.daily_twr >= 0 ? '+' : ''}
                        {(wealthData.daily_twr * 100).toFixed(2)}% (Daily)
                    </div>
                    <div className="mini-chart">
                        {wealthData.ratios.map((ratio, index) => (
                            <div key={index} className="sparkline-bar" style={{height: `${ratio}%`}}></div>
                        ))}
                    </div>
                </div>
            </div>

            {/* Market Pulse Card */}
            <div className="bento-card market-card">
                <div className="card-glass">
                    <span className="card-tag">MARKET PULSE</span>
                    <div className="market-scroll-container">
                        {marketPulseData.map((item, index) => (
                            <div key={index} className="market-item">
                                <span>{item.name}</span>
                                <span className="price">{item.latest_value.toLocaleString()}</span>
                                <span className={`change ${item.latest_twr >= 0 ? 'positive' : 'negative'}`}>
                                    {item.latest_twr >= 0 ? '+' : ''}{item.latest_twr}%
                                </span>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* System Status Card */}
            <div className="bento-card system-card">
                <div className="card-glass">
                    <span className="card-tag">SYSTEM CORE</span>
                    <div className="status-grid">
                        <div className="status-item">
                            <span className={`status-indicator ${apiOnline ? 'online' : 'offline'}`}></span>
                            <span>Backend API</span>
                        </div>
                        <div className="status-item">
                            <span className={`status-indicator ${dbOnline ? 'online' : 'offline'}`}></span>
                            <span>Database</span>
                        </div>
                        <div className="status-item">
                            <span className="status-indicator latency"></span>
                            <span>Latency: {latency !== null ? `${latency}ms` : '--'}</span>
                        </div>
                    </div>
                </div>
            </div>

            {/* Navigation Shortcuts Card */}
            <div className="bento-card shortcuts-card">
                <div className="card-glass">
                    <span className="card-tag">QUICK ACCESS</span>
                    <div className="shortcut-links">
                        <Link to="/portfolio" className="shortcut-item">📊 Performance</Link>
                        <Link to="/allocation_matrix" className="shortcut-item">🕸️ Allocation</Link>
                        <Link to="/ledger_navigator" className="shortcut-item">🧭 Ledger</Link>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Menu;
