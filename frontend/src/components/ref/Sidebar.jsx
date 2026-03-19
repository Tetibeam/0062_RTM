import React from 'react'
import { Link, useLocation } from 'react-router-dom'
import KPICommandCenter from './KPICommandCenter'
import KPIAllocationMatrix from './KPIAllocationMatrix'
import KPILedgerNavigator from './KPILedgerNavigator'
import KPIExtraordinaryEvents from './KPIExtraordinaryEvents'
import KPIInvestPerformanceLab from './KPIInvestPerformanceLab'
import KPIMarketRegimeRadar from './KPIMarketRegimeRadar'
import KPIRebalanceStrategyLab from './KPIRebalanceStrategyLab'
import KPIAssetDeepDrive from './KPIAssetDeepDrive'

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
          <Link to="/portfolio"><span className="nav-icon">📊</span> Portfolio Command Center</Link>
          <Link to="/allocation_matrix"><span className="nav-icon">🕸️</span> Allocation Matrix</Link>
          <Link to="/ledger_navigator"><span className="nav-icon">🧭</span> Ledger Navigator</Link>
          <Link to="/extraordinary_events"><span className="nav-icon">⚡</span> Extraordinary Events</Link>
          <Link to="/investment_performance_lab"><span className="nav-icon">🧪</span> Investment Performance Lab</Link>
          <Link to="/market_regime_radar"><span className="nav-icon">📡</span> Market Regime Radar</Link>
          <Link to="/rebalance_strategy_lab"><span className="nav-icon">⚖️</span> Rebalance Strategy Lab</Link>
          <Link to="/asset_deep_drive"><span className="nav-icon">🎯</span> Asset Deep Drive</Link>
        </nav>
      ) : (
        <div className="sidebar-content">
          {path === '/allocation_matrix' ? (
            <KPIAllocationMatrix />
          ) : path === '/ledger_navigator' ? (
            <KPILedgerNavigator />
          ) : path === '/extraordinary_events' ? (
            <KPIExtraordinaryEvents />
          ) : path === '/investment_performance_lab' ? (
            <KPIInvestPerformanceLab />
          ) : path === '/market_regime_radar' ? (
            <KPIMarketRegimeRadar />
          ) : path === '/rebalance_strategy_lab' ? (
            <KPIRebalanceStrategyLab
                value={selectedItems}
                onChange={setSelectedItems}/>
          ) : path === '/asset_deep_drive' ? (
            <KPIAssetDeepDrive 
                value={selectedItems}
                onChange={setSelectedItems}
            />
          ) : path === '/portfolio' ? (
            <KPICommandCenter />
          ) : (
            <div>Unknown Path: {path}</div>
          )}
        </div>
      )}
    </div>
  )
}

export default Sidebar
