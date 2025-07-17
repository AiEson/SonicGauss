# Model Switching Fix Documentation

## Problem
When switching between models in the SonicGauss webdemo, users encountered:
```
Uncaught SyntaxError: Identifier 'modelData' has already been declared (at data_4.js:1:1)
```

## Root Cause
Each data file (data.js, data_2.js, data_3.js, data_4.js) declares variables using `const`:
```javascript
const modelData = { /* ... */ };
const pointsData = { /* ... */ };
const audioData = { /* ... */ };
```

When dynamically loading new data files via script injection, JavaScript attempts to redeclare these `const` variables, causing a SyntaxError since `const` variables cannot be redeclared in the same scope.

## Solution
Replaced the dynamic script injection approach with a fetch-based solution that:

1. **Fetches data files as text** instead of executing them as scripts
2. **Modifies variable declarations** from `const` to `window.property` assignments
3. **Executes modified content** using Function constructor in controlled scope
4. **Allows variable reassignment** without redeclaration errors

### Key Changes

#### 1. Updated switchModel Function
```javascript
async function switchModel(modelFile) {
    try {
        // Fetch data file as text
        const response = await fetch(`./static/js/webdemo/${modelFile}`);
        const dataContent = await response.text();
        
        // Clear existing variables
        window.modelData = undefined;
        window.pointsData = undefined;
        window.audioData = undefined;
        
        // Transform const declarations to window assignments
        const modifiedContent = dataContent
            .replace(/^const modelData/m, 'window.modelData')
            .replace(/^const pointsData/m, 'window.pointsData')
            .replace(/^const audioData/m, 'window.audioData');
        
        // Execute in controlled scope
        const executeData = new Function(modifiedContent);
        executeData();
        
        // Load the model
        loadModelFromBase64();
        
    } catch (error) {
        console.error('Error switching models:', error);
        updateStatus(`Error loading model: ${error.message}`);
    }
}
```

#### 2. Updated Initial Data Loading
Modified the HTML initialization script to use the same fetch-based approach for consistency.

#### 3. Updated Variable References
Changed all references from `modelData`, `pointsData`, `audioData` to `window.modelData`, `window.pointsData`, `window.audioData` throughout the codebase.

## Benefits
- ✅ **No more redeclaration errors** - Variables can be reassigned without conflicts
- ✅ **Cleaner error handling** - Fetch API provides better error information than script injection
- ✅ **More reliable** - No dependency on script execution order or timing
- ✅ **Better debugging** - Clear error messages and stack traces
- ✅ **Consistent approach** - Same loading mechanism for initial and subsequent models

## Testing
Created `test_model_switching.html` to verify the fix works correctly across all model files.

## Files Modified
- `/static/js/webdemo/main.js` - Updated switchModel function and variable references
- `/index.html` - Updated initialization script to use fetch approach
- Added: `/test_model_switching.html` - Test page for verification

## Verification Steps
1. Open the main website (`index.html`)
2. Navigate to the Interactive Demo section
3. Try switching between different models using the dropdown
4. Verify no console errors occur
5. Confirm models load and display correctly
6. Test audio functionality by clicking white spheres

## Alternative Testing
Use the dedicated test page `test_model_switching.html` to systematically test model switching without the full 3D rendering context.