import React, { useRef, useEffect } from 'react';

export const MagnifyingGlass = ({
    referenceImage,
    testImage,
    position,
    zoomLevel = 3,
    magnifierSize = 200,
    visible = false,
    differenceMask = null
}) => {
    const refCanvasRef = useRef(null);
    const testCanvasRef = useRef(null);
    const refImgRef = useRef(null);
    const testImgRef = useRef(null);

    useEffect(() => {
        if (!referenceImage || !testImage) return;

        const refImg = new Image();
        refImg.onload = () => {
            refImgRef.current = refImg;
            updateMagnifier();
        };
        refImg.src = referenceImage;

        const testImg = new Image();
        testImg.onload = () => {
            testImgRef.current = testImg;
            updateMagnifier();
        };
        testImg.src = testImage;
    }, [referenceImage, testImage]);

    useEffect(() => {
        if (visible && position) {
            updateMagnifier();
        }
    }, [position, visible, differenceMask]);

    const updateMagnifier = () => {
        if (!refImgRef.current || !testImgRef.current || !position) return;

        drawMagnifiedView(refCanvasRef.current, refImgRef.current, false);
        drawMagnifiedView(testCanvasRef.current, testImgRef.current, true);
    };

    const drawMagnifiedView = (canvas, img, showDiff) => {
        if (!canvas || !img) return;

        const ctx = canvas.getContext('2d');
        canvas.width = magnifierSize;
        canvas.height = magnifierSize;

        // Calculate source rectangle (area to magnify)
        const sourceSize = magnifierSize / zoomLevel;
        const sx = Math.max(0, position.x - sourceSize / 2);
        const sy = Math.max(0, position.y - sourceSize / 2);
        const sw = Math.min(sourceSize, img.width - sx);
        const sh = Math.min(sourceSize, img.height - sy);

        // Draw magnified portion
        ctx.drawImage(
            img,
            sx, sy, sw, sh,
            0, 0, magnifierSize, magnifierSize
        );

        // Draw difference overlay if requested and available
        if (showDiff && differenceMask) {
            // Extract and scale the difference mask for this region
            const maskCanvas = document.createElement('canvas');
            const maskCtx = maskCanvas.getContext('2d');
            maskCanvas.width = differenceMask.width;
            maskCanvas.height = differenceMask.height;
            maskCtx.putImageData(differenceMask, 0, 0);

            ctx.globalAlpha = 0.4;
            ctx.drawImage(
                maskCanvas,
                sx, sy, sw, sh,
                0, 0, magnifierSize, magnifierSize
            );
            ctx.globalAlpha = 1.0;
        }

        // Draw crosshair
        ctx.strokeStyle = '#00ff00';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(magnifierSize / 2, 0);
        ctx.lineTo(magnifierSize / 2, magnifierSize);
        ctx.moveTo(0, magnifierSize / 2);
        ctx.lineTo(magnifierSize, magnifierSize / 2);
        ctx.stroke();
    };

    if (!visible || !position) {
        return null;
    }

    return (
        <div className="magnifying-glass-container">
            <div className="magnifier-pane">
                <h4>Reference (Zoomed {zoomLevel}x)</h4>
                <canvas ref={refCanvasRef} className="magnifier-canvas" />
            </div>
            <div className="magnifier-pane">
                <h4>Test (Zoomed {zoomLevel}x)</h4>
                <canvas ref={testCanvasRef} className="magnifier-canvas" />
            </div>
        </div>
    );
};
