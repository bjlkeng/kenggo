#include "game/boardhistory.h"

#include <iostream>
#include <stdexcept>
#include <string>

using Json = nlohmann::json;

namespace {

int boundedInteger(const Json& value, int minimum, int maximum, const std::string& name) {
  if(!value.is_number_integer() || value < minimum || value > maximum)
    throw std::invalid_argument(name + " must be an integer in [" + std::to_string(minimum) + "," + std::to_string(maximum) + "]");
  return value.get<int>();
}

void checkKeys(const Json& value, const std::set<std::string>& allowed, const std::string& name) {
  if(!value.is_object())
    throw std::invalid_argument(name + " must be an object");
  for(auto it = value.begin(); it != value.end(); ++it) {
    if(allowed.count(it.key()) == 0)
      throw std::invalid_argument(name + ": unknown field " + it.key());
  }
}

Player player(const Json& value) {
  if(value == "B") return P_BLACK;
  if(value == "W") return P_WHITE;
  throw std::invalid_argument("color must be B or W");
}

std::string playerName(Player value) {
  return value == P_BLACK ? "B" : "W";
}

Loc location(int action, int size) {
  return action == size * size ? Board::PASS_LOC : Location::getLoc(action % size, action / size, size);
}

Rules japaneseRules() {
  return Rules(Rules::KO_SIMPLE, Rules::SCORING_TERRITORY, Rules::TAX_SEKI,
               false, false, Rules::WHB_ZERO, false, 6.5f);
}

struct State {
  Board board;
  BoardHistory history;
  Player next;
  bool useHistory;
  int moveCount = 0;
  int passStreak = 0;

  State(const Board& initial, Player nextPlayer, bool historyMode)
    : board(initial),
      history(initial, nextPlayer, japaneseRules(), 0,
              BoardHistoryModes(/*alwaysComputePassAliveUnderSuicideRules=*/false,
                                /*excludeTerritoryAdjacentToAtari=*/true)),
      next(nextPlayer), useHistory(historyMode) {}

  bool legal(int action) const {
    if(useHistory && history.isGameFinished) return false;
    Loc loc = location(action, board.x_size);
    return useHistory ? history.isLegal(board, loc, next) : board.isLegal(loc, next, false);
  }
};

struct Queries {
  bool mask;
  std::vector<int> actions;
};

Json snapshot(const State& state, const Queries& queries) {
  const Board& board = state.board;
  const BoardHistory& history = state.history;
  const int size = board.x_size;
  Json rows = Json::array();
  for(int row = 0; row < size; ++row) {
    std::string text;
    for(int col = 0; col < size; ++col) {
      Color color = board.colors[Location::getLoc(col, row, size)];
      text += color == C_BLACK ? 'B' : color == C_WHITE ? 'W' : '.';
    }
    rows.push_back(text);
  }
  Json ko = nullptr;
  if(board.ko_loc != Board::NULL_LOC)
    ko = Location::getY(board.ko_loc, size) * size + Location::getX(board.ko_loc, size);
  const bool finished = state.useHistory && history.isGameFinished;
  const bool scored = finished && history.isScored;
  Json result = {
    {"board", rows}, {"to_play", playerName(state.next)}, {"simple_ko", ko},
    {"captures", {{"B", board.numBlackCaptures}, {"W", board.numWhiteCaptures}}},
    {"move_count", state.moveCount}, {"pass_streak", state.passStreak},
    {"encore_phase", state.useHistory ? history.encorePhase : 0},
    {"finished", finished}, {"no_result", finished && history.isNoResult}, {"scored", scored},
    {"winner", finished && history.winner != C_EMPTY ? Json(playerName(history.winner)) : Json(nullptr)},
    {"white_minus_black_halfpoints", scored ? Json(static_cast<int>(history.finalWhiteMinusBlackScore * 2)) : Json(nullptr)}
  };
  if(queries.mask) {
    Json mask = Json::array();
    for(int action = 0; action <= size * size; ++action) mask.push_back(state.legal(action));
    result["legal_mask"] = mask;
  }
  if(!queries.actions.empty()) {
    Json answers = Json::array();
    for(int action : queries.actions) answers.push_back({{"action", action}, {"legal", state.legal(action)}});
    result["query_legal"] = answers;
  }
  return result;
}

Json play(State& state, const Json& move, const Queries& queries) {
  Json result = {{"accepted", false}, {"reason", nullptr}};
  try {
    checkKeys(move, {"color", "action"}, "move");
    Player color = player(move.at("color"));
    int action = boundedInteger(move.at("action"), 0, state.board.x_size * state.board.y_size, "action");
    result["color"] = playerName(color);
    result["action"] = action;
    Loc loc = location(action, state.board.x_size);
    if(color != state.next) result["reason"] = "wrong_player";
    else if(state.useHistory && state.history.isGameFinished) result["reason"] = "finished";
    else if(!state.legal(action)) {
      if(state.board.colors[loc] != C_EMPTY) result["reason"] = "occupied";
      else if(state.board.isKoBanned(loc)) result["reason"] = "simple_ko";
      else if(state.board.isSuicide(loc, color)) result["reason"] = "suicide";
      else result["reason"] = "history_illegal";
    }
    else {
      if(state.useHistory) state.history.makeBoardMoveAssumeLegal(state.board, loc, color, nullptr);
      else state.board.playMoveAssumeLegal(loc, color);
      state.next = getOpp(color);
      state.moveCount++;
      state.passStreak = loc == Board::PASS_LOC ? state.passStreak + 1 : 0;
      result["accepted"] = true;
    }
  }
  catch(const std::exception& error) {
    result["reason"] = "malformed_move";
    result["error"] = error.what();
  }
  result["state"] = snapshot(state, queries);
  return result;
}

void checkMoves(const Json& moves) {
  if(!moves.is_array() || moves.size() > 2000)
    throw std::invalid_argument("moves must be an array with at most 2000 entries");
}

Json sequence(State& state, const Json& moves, const Queries& queries) {
  Json result = {{"initial", snapshot(state, queries)}, {"steps", Json::array()}};
  for(const Json& move : moves) result["steps"].push_back(play(state, move, queries));
  result["final"] = snapshot(state, queries);
  return result;
}

Json run(const Json& input) {
  checkKeys(input, {"size", "board", "to_play", "mode", "moves", "include_legal_mask",
                    "query_actions", "captures", "branches"}, "case");
  int size = boundedInteger(input.value("size", Json(19)), 2, 19, "size");
  Player next = player(input.value("to_play", Json("B")));
  std::string mode = input.value("mode", std::string("ordinary"));
  if(mode != "ordinary" && mode != "history") throw std::invalid_argument("mode must be ordinary or history");
  Board board(size, size);
  if(input.contains("board")) {
    const Json& rows = input.at("board");
    if(!rows.is_array() || rows.size() != static_cast<size_t>(size))
      throw std::invalid_argument("board must contain size rows");
    std::vector<Move> placements;
    for(int row = 0; row < size; ++row) {
      if(!rows[row].is_string()) throw std::invalid_argument("board rows must be strings");
      std::string cells = rows[row].get<std::string>();
      if(cells.size() != static_cast<size_t>(size)) throw std::invalid_argument("board rows must contain size cells");
      for(int col = 0; col < size; ++col) {
        char color = cells[col];
        if(color != '.' && color != 'B' && color != 'W') throw std::invalid_argument("board cells must be . B or W");
        if(color != '.') placements.emplace_back(Location::getLoc(col, row, size), color == 'B' ? P_BLACK : P_WHITE);
      }
    }
    if(!board.setStonesFailIfNoLibs(placements)) throw std::invalid_argument("setup contains a zero-liberty chain");
  }
  if(input.contains("captures")) {
    const Json& captures = input.at("captures");
    checkKeys(captures, {"B", "W"}, "captures");
    board.numBlackCaptures = boundedInteger(captures.at("B"), 0, 1000000, "captures.B");
    board.numWhiteCaptures = boundedInteger(captures.at("W"), 0, 1000000, "captures.W");
  }
  Queries queries{input.value("include_legal_mask", false), {}};
  if(input.contains("query_actions")) {
    const Json& actions = input.at("query_actions");
    if(!actions.is_array() || actions.size() > static_cast<size_t>(size * size + 1))
      throw std::invalid_argument("query_actions must be an array of at most size squared plus one actions");
    for(const Json& action : actions) queries.actions.push_back(boundedInteger(action, 0, size * size, "query action"));
  }
  Json moves = input.value("moves", Json::array());
  checkMoves(moves);
  Json branches = input.value("branches", Json::array());
  if(!branches.is_array() || branches.size() > 16) throw std::invalid_argument("branches must be an array of at most 16 move arrays");
  for(const Json& branch : branches) checkMoves(branch);
  State state(board, next, mode == "history");
  Json result = sequence(state, moves, queries);
  result["ok"] = true;
  result["mode"] = mode;
  result["size"] = size;
  if(!branches.empty()) {
    result["branches"] = Json::array();
    for(const Json& movesInBranch : branches) {
      State clone = state;
      result["branches"].push_back(sequence(clone, movesInBranch, queries));
    }
  }
  return result;
}

}  // namespace

int main() {
  Board::initHash();
  std::string line;
  while(std::getline(std::cin, line)) {
    try {
      if(line.size() > 4 * 1024 * 1024) throw std::invalid_argument("case exceeds 4 MiB");
      std::cout << run(Json::parse(line)).dump() << '\n';
    }
    catch(const std::exception& error) {
      std::cout << Json({{"ok", false}, {"error", error.what()}}).dump() << '\n';
    }
  }
  return std::cin.bad() ? 1 : 0;
}
