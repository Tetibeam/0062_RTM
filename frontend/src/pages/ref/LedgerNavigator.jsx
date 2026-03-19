import React, { useState, useEffect } from 'react'
import apiClient from '../apiClient'
import GraphContainer from '../components/GraphContainer'

function LedgerNavigator() {
  const [graphs, setGraphs] = useState({})
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    const fetchGraphs = async () => {
      try {
        const response = await apiClient.get('/Ledger_Navigator/graphs')
        setGraphs(response.data.graphs)
        setLoading(false)
      } catch (err) {
        console.error('Failed to load graphs:', err)
        setError(err.message)
        setLoading(false)
      }
    }

    fetchGraphs()
  }, [])

  // グラフの表示順
  const graphOrder = [
    'balanced_revenue_path',
    'investable_margin_trend',
    'liquidity_reserve',
    'spending_design_shift',
    'budget_integrity_matrix',
    'spending_vector'
  ]

  // グラフタイトルのマッピング
  const graphTitles = {
    'balanced_revenue_path': "<span><img src='/static/icon/star.svg' style='height:18px; margin-right:6px; opacity:0.85;'/> Balanced Revenue Path</span>",
    'investable_margin_trend': "<span><img src='/static/icon/budget.svg' style='height:18px; margin-right:6px; opacity:0.85;'/> Investable Margin Trend</span>",
    'liquidity_reserve': "<span><img src='/static/icon/waves.svg' style='height:18px; margin-right:6px; opacity:0.85;'/> Liquidity Reserve - Cash based</span>",
    'spending_design_shift': "<span><img src='/static/icon/line-chart.svg' style='height:20px; margin-right:6px; opacity:0.85;'/> Spending Design Shift</span>",
    'budget_integrity_matrix': "<span><img src='/static/icon/sail.svg' style='height:18px; margin-right:6px; opacity:0.85;'/> Budget Integrity Matrix</span>",
    'spending_vector': "<span><img src='/static/icon/line-chart.svg' style='height:20px; margin-right:6px; opacity:0.85;'/> Spending Vector</span>",
  }

  if (loading) {
    return <div className="main"><div>Loading graphs...</div></div>
  }

  if (error) {
    return <div className="main"><div>Error: {error}</div></div>
  }

  /* New: Handler for plot clicks */
  const handlePlotClick = (data, graphKey) => {
    if (data && data.points && data.points.length > 0) {
      const point = data.points[0];
      let label = null;

      if (graphKey === 'liquidity_reserve') {
        // Horizontal Bar chart: category is on y-axis
        label = point.y;
      } else if (graphKey === 'spending_vector') {
        // Scatter chart: item name is in customdata
        label = point.text;
      }

      if (label) {
        // Open new tab with standard 'label' parameter
        const url = `/ledger_navigator/${graphKey}/details?label=${encodeURIComponent(label)}`
        window.open(url, '_blank')
      }
    }
  }

  return (
    <div id="graphs-area" className="main">
      {graphOrder.map(key => {
        const figJson = graphs[key]
        if (!figJson) return null
        
        return (
          <GraphContainer
            key={key}
            figJson={figJson}
            titleHtml={graphTitles[key] || key}
            onPlotClick={(data) => handlePlotClick(data, key)}
          />
        )
      })}
    </div>
  )
}

export default LedgerNavigator
