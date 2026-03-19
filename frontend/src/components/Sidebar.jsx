import React from 'react'
import { Link, useLocation } from 'react-router-dom'
import KPIMarketRegimeRadar from './KPIMarketRegimeRadar'

function Sidebar({ selectedItems, setSelectedItems }) {
  const location = useLocation()
  const path = location.pathname.replace(/\/$/, '') || '/'

  const isMenu = path === '' || path === '/'

  return (
    <div className="sidebar">
      <div className="sidebar-top">
        <h1 className="sidebar-title">
          <span className="title-icon">🧭</span>
          <span className="title-text">Finance App</span>
        </h1>
        {!isMenu && (
          <Link to="/" className="back-button">
            <span className="back-icon">←</span> Back to Menu
          </Link>
        )}
      </div>

      {isMenu ? (
        <nav className="sidebar-nav">
          <Link to="/market_regime_radar"><span className="nav-icon">📡</span> Market Regime Radar</Link>
        </nav>
      ) : (
        <div className="sidebar-content">
          {path === '/market_regime_radar' ? (
            <KPIMarketRegimeRadar />
          ) : (
            <div>Unknown Path: {path}</div>
          )}
        </div>
      )}
    </div>
  )
}

export default Sidebar
