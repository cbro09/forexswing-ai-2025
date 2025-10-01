// Test the exact normalizePair function behavior
function normalizePair(pair) {
    // Convert to standard format (EUR/USD)
    const cleaned = pair.replace(/[^A-Z]/gi, '').toUpperCase();
    console.log(`Input: "${pair}" → Cleaned: "${cleaned}"`);
    if (cleaned.length === 6) {
        const result = `${cleaned.slice(0, 3)}/${cleaned.slice(3)}`;
        console.log(`Result: "${result}"`);
        return result;
    }
    console.log(`Fallback: "${pair}"`);
    return pair;
}

// Test with the exact input from the logs
const testInput = "EURUSD 15,";
console.log("Testing normalizePair with:", testInput);
const result = normalizePair(testInput);
console.log("Final result:", result);
console.log("URL encoded:", encodeURIComponent(result));