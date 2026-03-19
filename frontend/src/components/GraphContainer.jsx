import React, { useState, useRef, useEffect } from 'react'
import Plot from 'react-plotly.js'

function GraphContainer({ figJson, titleHtml, onPlotClick }) {
  const [isFullscreen, setIsFullscreen] = useState(false)
  const containerRef = useRef(null)
  const [containerSize, setContainerSize] = useState({ width: 0, height: 0 })

  // Stable ID for the plot div
  const plotIdRef = useRef(`plot-${Math.random().toString(36).substr(2, 9)}`)
  
  // Memoize fig parsing to ensure stable references
  const fig = React.useMemo(() => {
    return typeof figJson === 'string' ? JSON.parse(figJson) : figJson
  }, [figJson])
  
  // Measure container size
  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const updateSize = () => {
      const rect = container.getBoundingClientRect()
      
      // Get computed padding (1vh from CSS)
      const style = window.getComputedStyle(container)
      const paddingLeft = parseFloat(style.paddingLeft)
      const paddingRight = parseFloat(style.paddingRight)
      const paddingTop = parseFloat(style.paddingTop)
      const paddingBottom = parseFloat(style.paddingBottom)
      
      setContainerSize({
        width: rect.width - paddingLeft - paddingRight,
        height: rect.height - paddingTop - paddingBottom
      })
    }

    // Small delay to ensure DOM is fully rendered
    const timer = setTimeout(updateSize, 10)

    // Observe size changes
    const resizeObserver = new ResizeObserver(updateSize)
    resizeObserver.observe(container)

    return () => {
      clearTimeout(timer)
      resizeObserver.disconnect()
    }
  }, [isFullscreen])

  // Store default font sizes
  const layout = fig.layout || {}
  const defaultFonts = {
    font: layout.font?.size || 12,
    title: layout.title?.font?.size || 14,
    xaxis_title: layout.xaxis?.title?.font?.size || 12,
    yaxis_title: layout.yaxis?.title?.font?.size || 12,
    xaxis_tick: layout.xaxis?.tickfont?.size || 10,
    yaxis_tick: layout.yaxis?.tickfont?.size || 10,
    legend: layout.legend?.font?.size || 12
  }

  const [currentLayout, setCurrentLayout] = useState(fig.layout)
  // Ref to track the latest layout state from Plotly (including user zooms)
  const layoutRef = useRef(fig.layout)
  // Ref to track the last source figure to detect graph switches
  const lastFigRef = useRef(fig)

  // Update layout when container size, fullscreen state, or fig changes
  useEffect(() => {
    if (containerSize.width > 0 && containerSize.height > 0) {
      const titleHeight = 35 // Approximate height of title
      
      // Determine base layout:
      // If fig changed, reset to new fig.layout.
      // If only size/mode changed, use the latest known state (layoutRef) to preserve zooms.
      let baseLayout;
      if (fig !== lastFigRef.current) {
        baseLayout = fig.layout;
        lastFigRef.current = fig;
        // Reset ref to new base
        layoutRef.current = baseLayout; 
      } else {
        baseLayout = layoutRef.current || fig.layout;
      }

      // Calculate font scale
      let scale = 1
      if (isFullscreen) {
        scale = Math.min(window.innerWidth / 1000, 2.5) // Cap at 2.5x
      }

      const newLayout = {
        ...baseLayout,
        width: containerSize.width,
        height: containerSize.height - titleHeight,
        autosize: false,
        font: { ...baseLayout.font, size: defaultFonts.font * scale },
        title: { ...baseLayout.title, font: { size: defaultFonts.title * scale } },
        xaxis: { 
          ...baseLayout.xaxis, 
          title: { ...baseLayout.xaxis?.title, font: { size: defaultFonts.xaxis_title * scale } },
          tickfont: { ...baseLayout.xaxis?.tickfont, size: defaultFonts.xaxis_tick * scale }
        },
        yaxis: { 
          ...baseLayout.yaxis,
          title: { ...baseLayout.yaxis?.title, font: { size: defaultFonts.yaxis_title * scale } },
          tickfont: { ...baseLayout.yaxis?.tickfont, size: defaultFonts.yaxis_tick * scale }
        },
        legend: { ...baseLayout.legend, font: { size: defaultFonts.legend * scale } },
        uirevision: 'true' // Still keep this as backup
      };

      setCurrentLayout(newLayout);
      // Update ref immediately so subsequent unrelated updates use this foundation?
      // No, wait for onUpdate to confirm what Plotly did, but we can optimistically update.
      // Actually layoutRef should primarily capture user interactions. 
      // Merging calculated props into layoutRef might be circular if we aren't careful, 
      // but here we are producing the input for the next render.
    }
  }, [containerSize, isFullscreen, fig]) // Depend on 'fig' (stable), not 'fig.layout'

  const handleTitleClick = () => {
    setIsFullscreen(!isFullscreen)
  }



  return (
    <div className={`graph-container ${isFullscreen ? 'graph-fullscreen' : ''}`} ref={containerRef}>


      <div
        className="graph-title" 
        dangerouslySetInnerHTML={{ __html: titleHtml }}
        onClick={handleTitleClick}
        style={{ cursor: 'pointer' }}
      />

      {containerSize.width > 0 && (
        <Plot
          divId={plotIdRef.current}
          data={fig.data}
          layout={currentLayout}
          onUpdate={(figure) => { layoutRef.current = figure.layout }}
          config={{
            responsive: false,
            displayModeBar: false
          }}
          onClick={onPlotClick}
          style={{ 
            width: `${containerSize.width}px`, 
            height: `${containerSize.height - 35}px` 
          }}
        />
      )}
    </div>
  )
}

export default GraphContainer
