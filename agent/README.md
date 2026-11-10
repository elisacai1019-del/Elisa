# Elisa opportunity agent

Run the service from this directory with `npm start`. It listens on port 8080 by
default and its health check is `GET /`, which returns `agent is running`.

The service reads `OPENAI_API_KEY` and `AGENT_SECRET` from the environment, plus
the local `interests.txt` profile when a research job starts.

For a direct Railway upload, run `railway up` from this directory and select
the existing `agent` service.
