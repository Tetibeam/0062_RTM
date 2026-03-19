import React, { useState, useEffect } from 'react'
import apiClient from '../apiClient'
import GraphContainer from '../components/GraphContainer'

function MarketRegimeRadar() {
  const [graphs, setGraphs] = useState({})
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    const fetchGraphs = async () => {
      try {
        const response = await apiClient.get('/Market_Regime_Radar/graphs')

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
    'regime_transitions',
    'early_warning',
    'regime_snapshot',
    'decision_confidence',
    'rolling_driver_path',
    'regime_playbook',
  ]

  // グラフタイトルのマッピング
  const graphTitles = {
    'regime_transitions': "<span><img src='/static/icon/lighthouse.svg' style='height:18px; margin-right:6px; opacity:0.85;'/> Regime Transitions</span>",
    'decision_confidence': "<span><img src='/static/icon/lighthouse.svg' style='height:18px; margin-right:6px; opacity:0.85;'/> Decision Confidence</span>",
    'early_warning': "<span><img src='/static/icon/lighthouse.svg' style='height:18px; margin-right:6px; opacity:0.85;'/> Early Warning</span>",
    'rolling_driver_path': "<span><img src='/static/icon/lighthouse.svg' style='height:18px; margin-right:6px; opacity:0.85;'/> Rolling Driver Path</span>",
    'regime_snapshot': "<span><img src='/static/icon/lighthouse.svg' style='height:18px; margin-right:6px; opacity:0.85;'/> Regime Snapshot</span>",
    'regime_playbook': "<span><img src='/static/icon/star.svg' style='height:18px; margin-right:6px; opacity:0.85;'/> Regime Playbook</span>"
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

export default MarketRegimeRadar
