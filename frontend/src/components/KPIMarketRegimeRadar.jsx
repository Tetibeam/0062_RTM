import React, { useState, useEffect } from 'react'
import axios from 'axios'
import Plot from 'react-plotly.js'

function KPIMarketRegimeRadar() {
  return (
    <div id="dashboard-summary">
      <div className="kpi-header">
        <h3 className="kpi-title">📡 Market Regime Radar</h3>
      </div>
      <div className="summary-grid">
        <div style={{gridColumn: 'span 2', fontSize: '1.4vh', color: '#666', marginTop: '1vh'}}>
          Market analysis pending...
        </div>
      </div>
    </div>
  )
}

export default KPIMarketRegimeRadar
