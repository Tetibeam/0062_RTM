import React, { useState, useEffect } from 'react'
import apiClient from '../apiClient'

function KPIRebalanceStrategyLab({value = [], onChange = () => {}}) {
    const [options, setOptions] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        const fetchSummary = async () => {
            try {
                const response = await apiClient.get('/Rebalance_Strategy_Lab/summary')
                const summaryData = Array.isArray(response.data.summary) ? response.data.summary : [];
                setOptions(summaryData)
                setLoading(false)
            } catch (err) {
                console.error('Failed to load dashboard summary:', err)
                setError(err.message)
                setLoading(false)
            }
        }

        fetchSummary()
    }, [])

    if (loading) return <div id="dashboard-summary">Loading...</div>
    if (error) return <div id="dashboard-summary">Error: {error}</div>

    return (
        <div id="dashboard-summary">
            <div className="kpi-header">
                <h3 className="kpi-title">⚖️ Rebalance Strategy Lab</h3>
            </div>

            <select
                multiple
                size={options.length > 0 ? Math.min(Math.max(options.length, 5), 20) : 5}
                value={value}
                onChange={(e) =>
                    onChange(
                        Array.from(e.target.selectedOptions, (o) => o.value)
                    )
                }
                style={{ 
                    width: "100%", 
                    backgroundColor: "#1f1f1f", 
                    color: "#e0e0e0", 
                    border: "1px solid #333",
                    borderRadius: "4px",
                    padding: "4px",
                    fontSize: "2vh",
                    fontFamily: "'Montserrat', sans-serif"
                }}
            >
                {options.length > 0 ? (
                    options.map((opt) => (
                        <option key={opt} value={opt}>
                            {opt}
                        </option>
                    ))
                ) : (
                    <option disabled>No asset classes found</option>
                )}
            </select>
        </div>
    );
}


export default KPIRebalanceStrategyLab
