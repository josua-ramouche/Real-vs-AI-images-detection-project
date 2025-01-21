import { AppBar, Toolbar, Typography, Box } from "@mui/material";
import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom";
import './App.css';
import Page from "./components/Page";
import Demo from "./components/Demo";
import ResultsPage from "./components/ResultsPage";
import InfoAnalysePage from "./components/InfoAnalysePage";

function App() {
  return (
    <Router>
      <div>
        {/* Barre de navigation */}
        <AppBar position="fixed" sx={{ backgroundColor: '#65b2a0' }}>
          <Toolbar>
            <Typography
              variant="h5"
              component="div"
              sx={{ flexGrow: 1, textAlign: 'center', fontWeight: 'bold' }}
            >
              Real vs AI Generated Image
            </Typography>
          </Toolbar>
        </AppBar>

        {/* Contenu principal */}
        <Box sx={{ paddingTop: '64px' }}>
          <Routes>
            <Route path="/" element={<Navigate to="/home" />} />
            <Route path="/home" element={<Page />} />
            <Route path="/demo" element={<Demo />} />
            <Route path="/results" element={<ResultsPage />} />
            <Route path="/info" element={<InfoAnalysePage />} />
          </Routes>
        </Box>
      </div>
    </Router>
  );
}

export default App;
