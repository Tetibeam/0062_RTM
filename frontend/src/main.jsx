import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import App from './App.jsx'
import './styles/style.css'

if (import.meta.env.DEV && import.meta.env.VITE_USE_MSW === 'true') {
  const { worker } = await import('./mocks/browser')
  worker.start({
    onUnhandledRequest: 'warn',
  })
}

console.log("ReactDOM.render execution start")
ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </React.StrictMode>,
)
