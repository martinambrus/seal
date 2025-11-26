import React, { useState, useEffect } from 'react';
import { ImageUploader } from './components/ImageUploader';
import { ImageViewer } from './components/ImageViewer';
import { MagnifyingGlass } from './components/MagnifyingGlass';
import { MethodControls } from './components/MethodControls';
import { ViewModeSelector } from './components/ViewModeSelector';
import { ResultsPanel } from './components/ResultsPanel';
import { api } from './services/api';
import './App.css';

function App() {
    // Image state
    const [referenceFile, setReferenceFile] = useState(null);
    const [testFile, setTestFile] = useState(null);
    const [referencePreview, setReferencePreview] = useState(null);
    const [testPreview, setTestPreview] = useState(null);

    // Configuration state
    const [config, setConfig] = useState(null);
    const [enabledMethods, setEnabledMethods] = useState({
        ssim: true,
        pixel_diff: true,
        color: true,
        edge: true,
        ocr: true,
        siamese: false,
        cnn: false,
        autoencoder: false
    });
    const [thresholds, setThresholds] = useState({
        ssim: 0.95,
        pixel_diff: 0.20,
        color: 0.25,
        edge: 0.15,
        ocr: 0.80,
        siamese: 0.90,
        cnn: 0.75,
        autoencoder: 0.30
    });
    const [displayThreshold, setDisplayThreshold] = useState(0.30);

    // Results state
    const [results, setResults] = useState(null);
    const [processing, setProcessing] = useState(false);
    const [error, setError] = useState(null);

    // View state
    const [viewMode, setViewMode] = useState('combined');
    const [currentDifferenceOverlay, setCurrentDifferenceOverlay] = useState(null);

    // Magnifier state
    const [magnifierPosition, setMagnifierPosition] = useState(null);
    const [magnifierVisible, setMagnifierVisible] = useState(false);
    const [magnifierZoom, setMagnifierZoom] = useState(3);

    // Load default configuration on mount
    useEffect(() => {
        loadDefaultConfig();
    }, []);

    const loadDefaultConfig = async () => {
        try {
            const defaultConfig = await api.getDefaultConfig();
            setConfig(defaultConfig);
        } catch (err) {
            console.error('Failed to load default config:', err);
            // Use hardcoded defaults if API fails
            setConfig({
                alignment: { method: 'orb', min_quality: 0.7 },
                upsampling: { enabled: false, method: 'bicubic', factor: 2 },
                methods: {
                    ssim: { enabled: true, threshold: 0.95 },
                    pixel_diff: { enabled: true, threshold: 0.20 },
                    color: { enabled: true, threshold: 0.25 },
                    edge: { enabled: true, threshold: 0.15 },
                    ocr: { enabled: true, threshold: 0.80 },
                    siamese: { enabled: false, threshold: 0.90 },
                    cnn: { enabled: false, threshold: 0.75 },
                    autoencoder: { enabled: false, threshold: 0.30 }
                },
                display: { threshold: displayThreshold, view_mode: 'combined' }
            });
        }
    };

    const handleReferenceImageSelected = (file, preview) => {
        setReferenceFile(file);
        setReferencePreview(preview);
        setResults(null); // Clear previous results
    };

    const handleTestImageSelected = (file, preview) => {
        setTestFile(file);
        setTestPreview(preview);
        setResults(null);
    };

    const handleMethodToggle = (methodName, enabled) => {
        setEnabledMethods(prev => ({
            ...prev,
            [methodName]: enabled
        }));
    };

    const handleThresholdChange = (methodName, value) => {
        setThresholds(prev => ({
            ...prev,
            [methodName]: value
        }));
    };

    const handleDisplayThresholdChange = (value) => {
        setDisplayThreshold(value);
    };

    const handleViewModeChange = (mode) => {
        setViewMode(mode);
        updateDifferenceOverlay(mode, results);
    };

    const handleCompare = async () => {
        if (!referenceFile || !testFile) {
            setError('Please upload both reference and test images');
            return;
        }

        setProcessing(true);
        setError(null);

        try {
            // Build configuration
            const comparisonConfig = {
                alignment: config?.alignment || { method: 'orb', min_quality: 0.7 },
                upsampling: config?.upsampling || { enabled: false, method: 'bicubic', factor: 2 },
                methods: Object.fromEntries(
                    Object.entries(enabledMethods).map(([method, enabled]) => [
                        method,
                        { enabled, threshold: thresholds[method] }
                    ])
                ),
                display: {
                    threshold: displayThreshold,
                    view_mode: viewMode
                }
            };

            // Call API
            const result = await api.compareStamps(referenceFile, testFile, comparisonConfig);

            if (result.success) {
                setResults(result);
                updateDifferenceOverlay(viewMode, result);
            } else {
                setError(result.error || 'Comparison failed');
            }
        } catch (err) {
            console.error('Comparison error:', err);
            setError(`Failed to compare images: ${err.message}`);
        } finally {
            setProcessing(false);
        }
    };

    const updateDifferenceOverlay = (mode, resultData) => {
        if (!resultData || !resultData.results) {
            setCurrentDifferenceOverlay(null);
            return;
        }

        // Extract appropriate difference map based on view mode
        let differenceMap = null;

        if (mode === 'combined' && resultData.results.ensemble) {
            differenceMap = resultData.results.ensemble.difference_map;
        } else if (resultData.results[mode]) {
            differenceMap = resultData.results[mode].difference_map;
        }

        // Store the raw data (would need conversion to ImageData for actual overlay)
        setCurrentDifferenceOverlay(differenceMap);
    };

    const handleMouseMove = (x, y) => {
        setMagnifierPosition({ x, y });
        setMagnifierVisible(true);
    };

    const handleMouseLeave = () => {
        setMagnifierVisible(false);
    };

    const handleRegionClick = (region) => {
        // Center magnifier on clicked region
        const centerX = region.bbox[0] + region.bbox[2] / 2;
        const centerY = region.bbox[1] + region.bbox[3] / 2;
        setMagnifierPosition({ x: centerX, y: centerY });
        setMagnifierVisible(true);
    };

    const getMethodStats = () => {
        if (!results || !results.results) return {};

        const stats = {};
        Object.entries(results.results).forEach(([method, data]) => {
            if (data) {
                stats[method] = {
                    regions: data.num_regions || 0,
                    confidence: Math.round((data.overall_score || 0) * 100)
                };
            }
        });

        return stats;
    };

    const getAvailableMethods = () => {
        if (!results || !results.results) return [];
        return Object.keys(results.results).filter(key => results.results[key]);
    };

    return (
        <div className="app">
            <header className="app-header">
                <h1>🔍 Stamp Comparison Tool</h1>
                <p>Advanced image analysis for detecting stamp variants and discrepancies</p>
            </header>

            <div className="app-container">
                {/* Left Panel: Image Upload */}
                <div className="upload-panel">
                    <ImageUploader
                        label="Reference Stamp"
                        onImageSelected={handleReferenceImageSelected}
                        currentImage={referencePreview}
                    />
                    <ImageUploader
                        label="Test Stamp"
                        onImageSelected={handleTestImageSelected}
                        currentImage={testPreview}
                    />

                    <button
                        className="compare-button"
                        onClick={handleCompare}
                        disabled={!referenceFile || !testFile || processing}
                    >
                        {processing ? '⏳ Processing...' : '🔍 Compare Stamps'}
                    </button>

                    {error && (
                        <div className="error-message">
                            <span>⚠️ {error}</span>
                        </div>
                    )}

                    {results && results.alignment_quality !== undefined && (
                        <div className="alignment-info">
                            <h4>Alignment Quality</h4>
                            <div className="quality-bar">
                                <div
                                    className="quality-fill"
                                    style={{ width: `${results.alignment_quality}%` }}
                                />
                            </div>
                            <span>{Math.round(results.alignment_quality)}%</span>
                        </div>
                    )}
                </div>

                {/* Center Panel: Image Viewers */}
                <div className="viewer-panel">
                    <div className="viewers-container">
                        <ImageViewer
                            image={referencePreview}
                            title="Reference Stamp"
                            width={400}
                            height={400}
                            onMouseMove={handleMouseMove}
                            onMouseLeave={handleMouseLeave}
                        />

                        <ImageViewer
                            image={testPreview}
                            title="Test Stamp"
                            width={400}
                            height={400}
                            onMouseMove={handleMouseMove}
                            onMouseLeave={handleMouseLeave}
                            overlayMask={currentDifferenceOverlay}
                        />
                    </div>

                    {results && (
                        <div className="view-controls">
                            <div className="threshold-control-main">
                                <label>
                                    Display Threshold: <strong>{displayThreshold.toFixed(2)}</strong>
                                    <input
                                        type="range"
                                        min="0"
                                        max="1"
                                        step="0.01"
                                        value={displayThreshold}
                                        onChange={(e) => handleDisplayThresholdChange(parseFloat(e.target.value))}
                                        className="threshold-slider-main"
                                    />
                                </label>
                            </div>

                            <div className="magnifier-controls">
                                <label>
                                    Magnifier Zoom: <strong>{magnifierZoom}x</strong>
                                    <input
                                        type="range"
                                        min="2"
                                        max="6"
                                        step="1"
                                        value={magnifierZoom}
                                        onChange={(e) => setMagnifierZoom(parseInt(e.target.value))}
                                        className="zoom-slider"
                                    />
                                </label>
                            </div>
                        </div>
                    )}

                    {magnifierVisible && referencePreview && testPreview && (
                        <MagnifyingGlass
                            referenceImage={referencePreview}
                            testImage={testPreview}
                            position={magnifierPosition}
                            zoomLevel={magnifierZoom}
                            magnifierSize={200}
                            visible={magnifierVisible}
                            differenceMask={currentDifferenceOverlay}
                        />
                    )}
                </div>

                {/* Right Panel: Controls and Results */}
                <div className="control-panel">
                    <MethodControls
                        enabledMethods={enabledMethods}
                        thresholds={thresholds}
                        onMethodToggle={handleMethodToggle}
                        onThresholdChange={handleThresholdChange}
                        processingStatus={processing ? { all: 'processing' } : {}}
                    />

                    {results && (
                        <>
                            <ViewModeSelector
                                currentView={viewMode}
                                onViewChange={handleViewModeChange}
                                availableMethods={getAvailableMethods()}
                                methodStats={getMethodStats()}
                            />

                            <ResultsPanel
                                results={results}
                                currentView={viewMode}
                                onRegionClick={handleRegionClick}
                            />
                        </>
                    )}
                </div>
            </div>

            <footer className="app-footer">
                <p>Powered by advanced computer vision and machine learning</p>
            </footer>
        </div>
    );
}

export default App;
