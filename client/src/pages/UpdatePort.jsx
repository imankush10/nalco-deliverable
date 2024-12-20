import React, { useState } from "react";

function UpdatePort() {
  const [port, setPort] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await fetch(`http://localhost:8000/set-port/${port}`, {
        method: "POST",
      });
      const data = await response.json();
      alert(data.message);
    } catch (error) {
      alert("Failed to update port. Check the backend.");
    }
  };

  return (
    <div>
      <form onSubmit={handleSubmit}>
        <label>
          Enter SQL Server Port:
          <input
            type="number"
            value={port}
            onChange={(e) => setPort(e.target.value)}
            required
          />
        </label>
        <button type="submit">Update Port</button>
      </form>
    </div>
  );
}

export default UpdatePort;
