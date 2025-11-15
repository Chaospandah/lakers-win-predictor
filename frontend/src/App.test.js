import { act, render, screen, waitFor } from '@testing-library/react';
import App from './App';

beforeEach(() => {
  global.fetch = jest.fn();
});

afterEach(() => {
  jest.resetAllMocks();
});

test('fetches prediction data and renders it', async () => {
  const mockNextGame = {
    opponent: 'BOS',
    opponent_id: 1,
    game_date: '2025-11-20',
    home: true,
    win_probability: 0.75,
    prediction: 1,
  };

  global.fetch.mockResolvedValueOnce({
    ok: true,
    json: async () => mockNextGame,
  });

  await act(async () => {
    render(<App />);
  });

  expect(screen.getByText(/Lakers Win Predictor/i)).toBeInTheDocument();

  await waitFor(() => expect(screen.getByText(/model pick/i)).toBeInTheDocument());

  expect(screen.getAllByText(/Lakers Win/i).length).toBeGreaterThan(0);
  expect(screen.getByText(/Opponent/i)).toBeInTheDocument();
  expect(screen.getByText(/BOS/i)).toBeInTheDocument();
  expect(screen.getAllByText(/Date/i).length).toBeGreaterThan(0);
  expect(global.fetch).toHaveBeenCalledTimes(1);
  expect(global.fetch).toHaveBeenCalledWith(expect.stringContaining('/next-game-prediction'));
});
