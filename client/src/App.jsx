import React from 'react';
import CameraComponent from './components/Camera';

function App() {
  return (
    <div className="App">
      <h1>Tacit v2: Vision Test</h1>
      <div style={{ display: 'flex', justifyContent: 'center', marginTop: '20px' }}>
        <CameraComponent />
      </div>
    </div>
  );
}

export default App;