import React, { useState, useEffect } from 'react';
import { api } from './services/api';
import './App.css';

function App() {
    const [referenceImage, setReferenceImage] = useState(null);
    const [testImage, setTestImage] = useState(null);
    const [referencePreview, setReferencePreview] = useState(null);
    const [testPreview, setTestPreview] = useState(null);
    const [config, setConfig] = useState(null);
    const [results, setResults] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [backendStatus, setBackendStatus] = useState('checking');

    useEffect(() => {
        // Load default config on mount
        loadDefaultConfig();
        checkBackendHealth();
    }, []);

    const checkBackendHealth = async () => {
        try {
            await api.healthCheck();
            setBackendStatus('connected');
        } catch (err) {
            setBackendStatus('disconnected');
        }
    };

    const loadDefaultConfig = async () => {
        try {
            const defaultConfig = await api.getDefaultConfig();
            setConfig(defaultConfig);
        } catch (err) {
            console.error('Failed to load default config:', err);
            setError('Failed to load configuration');
        }
    };

    const handleImageUpload = (file, type) => {
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (e) => {
            if (type === 'reference') {
                setReferenceImage(file);
                setReferencePreview(e.target.result);
            } else {
                setTestImage(file);
                setTestPreview(e.target.result);
            }
        };
        reader.readAsDataURL(file);
    };

    const handleCompare = async () => {
        if (!referenceImage || !testImage || !config) {
            setError('Please upload both images');
            return;
        }

        setLoading(true);
        setError(null);
        setResults(null);

        try {
            const result = await api.compareStamps(referenceImage, testImage, config);

            if (result.success) {
                setResults(result);
            } else {
                setError(result.error || 'Comparison failed');
            }
        } catch (err) {
            console.error('Comparison error:', err);
            setError(err.response?.data?.error || err.message || 'Comparison failed');
        } finally {
            setLoading(false);
        }
    };

    const toggleMethod = (methodName) => {
        if (!config) return;

        setConfig({
            ...config,
            methods: {
                ...config.methods,
                [methodName]: {
                    ...config.methods[methodName],
                    enabled: !config.methods[methodName].enabled
                }
            }
        });
    };

    return (
        <div className="App">
            <header className="app-header">
                <h1>🔍 Stamp Comparator</h1>
                <div className="status-indicator">
                    Backend: <span className={`status-${backendStatus}`}>{backendStatus}</span>
                </div>
            </header>

            <div className="main-container">
                {/* Upload Section */}
                <div className="upload-section">
                    <div className="upload-box">
                        <h3>Reference Image</h3>
                        <input
                            type="file"
                            accept="image/*"
                            onChange={(e) => handleImageUpload(e.target.files[0], 'reference')}
                            id="reference-upload"
                        />
                        <label htmlFor="reference-upload" className="upload-label">
                            {referencePreview ? (
                                <img src={referencePreview} alt="Reference" className="preview-image" />
                            ) : (
                                <div className="upload-placeholder">
                                    <span>📁 Upload Reference</span>
                                </div>
                            )}
                        </label>
                    </div>

                    <div className="upload-box">
                        <h3>Test Image</h3>
                        <input
                            type="file"
                            accept="image/*"
                            onChange={(e) => handleImageUpload(e.target.files[0], 'test')}
                            id="test-upload"
                        />
                        <label htmlFor="test-upload" className="upload-label">
                            {testPreview ? (
                                <img src={testPreview} alt="Test" className="preview-image" />
                            ) : (
                                <div className="upload-placeholder">
                                    <span>📁 Upload Test Image</span>
                                </div>
                            )}
                        </label>
                    </div>
                </div>

                {/* Controls Section */}
                {config && (
                    <div className="controls-section">
                        <h3>Detection Methods</h3>
                        <div className="method-toggles">
                            {Object.entries(config.methods).map(([methodName, methodConfig]) => (
                                <label key={methodName} className="method-toggle">
                                    <input
                                        type="checkbox"
                                        checked={methodConfig.enabled}
                                        onChange={() => toggleMethod(methodName)}
                                    />
                                    <span className="method-name">{methodName.toUpperCase()}</span>
                                </label>
                            ))}
                        </div>

                        <button
                            className="compare-button"
                            onClick={handleCompare}
                            disabled={loading || !referenceImage || !testImage}
                        >
                            {loading ? 'Comparing...' : 'Compare Images'}
                        </button>
                    </div>
                )}

                {/* Error Display */}
                {error && (
                    <div className="error-message">
                        ⚠️ {error}
                    </div>
                )}

                {/* Results Section */}
                {results && (
                    <div className="results-section">
                        <h2>Comparison Results</h2>

                        <div className="results-summary">
                            <div className="result-card">
                                <h4>Alignment Quality</h4>
                                <div className="result-value">{results.alignment_quality.toFixed(2)}%</div>
                            </div>

                            <div className="result-card">
                                <h4>Execution Time</h4>
                                <div className="result-value">{results.execution_time.toFixed(2)}s</div>
                            </div>

                            {results.results.ensemble && (
                                <div className="result-card">
                                    <h4>Differences Found</h4>
                                    <div className="result-value">{results.results.ensemble.num_regions}</div>
                                </div>
                            )}
                        </div>

                        <div className="method-results">
                            <h3>Method Results</h3>
                            {Object.entries(results.results).map(([methodName, methodResult]) => (
                                methodName !== 'ensemble' && methodResult && (
                                    <div key={methodName} className="method-result-card">
                                        <h4>{methodName.toUpperCase()}</h4>
                                        <div className="method-stats">
                                            <span>Score: {methodResult.overall_score?.toFixed(3)}</span>
                                            <span>Regions: {methodResult.num_regions}</span>
                                            <span>Time: {methodResult.execution_time?.toFixed(2)}s</span>
                                        </div>
                                    </div>
                                )
                            ))}
                        </div>

                        {results.results.ensemble && (
                            <div className="ensemble-results">
                                <h3>Ensemble Analysis</h3>
                                <div className="ensemble-stats">
                                    <p><strong>Overall Confidence:</strong> {(results.results.ensemble.overall_confidence * 100).toFixed(1)}%</p>
                                    <p><strong>Methods Used:</strong> {results.results.ensemble.num_methods_used}</p>
                                    <p><strong>Total Regions:</strong> {results.results.ensemble.num_regions}</p>
                                </div>

                                {results.results.ensemble.regions.length > 0 && (
                                    <div className="regions-list">
                                        <h4>Detected Regions</h4>
                                        {results.results.ensemble.regions.slice(0, 5).map((region, idx) => (
                                            <div key={idx} className="region-item">
                                                <span>Region {idx + 1}</span>
                                                <span>Confidence: {(region.confidence * 100).toFixed(1)}%</span>
                                                <span>Area: {region.area_pixels}px</span>
                                                <span>Detected by: {region.detected_by.join(', ')}</span>
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}

export default App;
