import {
  BrowserRouter as Router,
  Routes,
  Route,
  Navigate,
} from "react-router-dom";
import PredictPage from "./pages/PredictPage";
import Layout from "./components/Layout";
import ReversePage from "./pages/ReversePage";
import TrainPage from "./pages/TrainPage";
import DashboardPage from "./pages/DashboardPage";
import HomePage from "./pages/HomePage";
import DefaultPage from "./pages/DefaultPage";
import UpdatePort from "./pages/UpdatePort";

function App() {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route index element={<Navigate to="/home" />} />
          <Route path="/home" element={<HomePage />} />
          <Route path="/predict" element={<PredictPage />} />
          <Route path="/reverse" element={<ReversePage />} />
          <Route path="/train" element={<TrainPage />} />
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/default-values" element={<DefaultPage />} />
          <Route path="/port" element={<UpdatePort />} />
        </Routes>
      </Layout>
    </Router>
  );
}

export default App;
