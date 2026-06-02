import { NextResponse } from "next/server";

const apiUrl = process.env.PREDICTION_API_URL ?? "http://127.0.0.1:8000";

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const response = await fetch(`${apiUrl}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      cache: "no-store",
    });

    const data = await response.json();
    return NextResponse.json(data, { status: response.status });
  } catch {
    return NextResponse.json(
      {
        predictions: [],
        errors: [
          {
            ticker: "API",
            message:
              "Prediction service is unavailable. Start the FastAPI server on port 8000.",
          },
        ],
      },
      { status: 503 },
    );
  }
}
