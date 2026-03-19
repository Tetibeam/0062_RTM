import React from 'react'
import { Routes, Route } from 'react-router-dom'
import Sidebar from './components/Sidebar'
import MarketRegimeRadar from './pages/MarketRegimeRadar'
import Menu from './pages/Menu'


function App() {
  const [selectedItems, setSelectedItems] = React.useState([]);

  return (
    <div className="app-container">
      <Sidebar
        selectedItems={selectedItems}
        setSelectedItems={setSelectedItems}
      />
      <Routes>
        <Route path="/" element={<Menu />} />

        {/* market_regime_radar */}
        <Route path="/market_regime_radar" element={<MarketRegimeRadar />} />

        {/*<Route path="/rebalance_strategy_lab" element={<RebalanceStrategyLab selectedItems={selectedItems} />} />*/}

      </Routes>
    </div>
  )
}
export default App
