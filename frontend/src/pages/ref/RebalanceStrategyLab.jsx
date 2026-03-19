import React, { useState, useEffect } from 'react'
import apiClient from '../apiClient'
import GraphContainer from '../components/GraphContainer'

function RebalanceStrategyLab({ selectedItems = [] }) {
  const [graphs, setGraphs] = useState({})
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    const fetchGraphs = async () => {
      try {
        setLoading(true)
        const response = await apiClient.post('/Rebalance_Strategy_Lab/graphs', {
            selected_items: selectedItems
        })
        setGraphs(response.data.graphs)
  
        setLoading(false)
      } catch (err) {
        console.error('Failed to load graphs:', err)
        setError(err.message)
        setLoading(false)
      }
    }

    fetchGraphs()
  }, [selectedItems])

  // グラフの表示順
  const graphOrder = [
    'equity_market_resilience',
    'equity_mid_term_resilience',
    'shock_resilience',
    'risk_on_entry',
    'regime_transition_response',
    'recovery_power'
  ]

  // グラフタイトルのマッピング
  const graphTitles = {
    'equity_market_resilience': "<span><img src='/static/icon/star.svg' style='height:18px; margin-right:6px; opacity:0.85;'/> Equity Market Resilience</span>",
    'shock_resilience': "<span><img src='/static/icon/star.svg' style='height:18px; margin-right:6px; opacity:0.85;'/> Shock Resilience</span>",
    'equity_mid_term_resilience': "<span><img src='/static/icon/star.svg' style='height:18px; margin-right:6px; opacity:0.85;'/> Equity Mid Term Resilience</span>",
    'risk_on_entry': "<span><img src='/static/icon/sail.svg' style='height:18px; margin-right:6px; opacity:0.85;'/> Risk On Entry</span>",
    'regime_transition_response': "<span><img src='/static/icon/compass.svg' style='height:20px; margin-right:6px; opacity:0.85;'/> Regime Transition Response</span>",
    'recovery_power': "<span><img src='/static/icon/line-chart.svg' style='height:20px; margin-right:6px; opacity:0.85;'/> Recovery Power</span>"
  }

  if (loading) {
    return <div className="main"><div>Loading graphs...</div></div>
  }

  if (error) {
    return <div className="main"><div>Error: {error}</div></div>
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
          />
        )
      })}
    </div>
  )
}

export default RebalanceStrategyLab