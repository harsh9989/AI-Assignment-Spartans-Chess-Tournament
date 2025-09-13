# B22ES027.py
import sys
import copy
import time
from config import *
from board import Board

class B22ES027:
    """
    Alpha-beta adversarial agent with iterative deepening, transposition table,
    simple move ordering and a heuristic evaluation. Designed to be dropped
    into the assignment template. Adjust small adapter methods below if your
    Board API uses different method names.
    """

    def __init__(self, board):
        self.board = board
        self.nodes_expanded = 0
        self.depth = 3           # default search depth (can be changed)
        self.time_limit = 0.9    # fraction of allowed time to use for iterative deepening (if using timing)
        self.start_time = None
        self.max_time = None
        self.transposition_table = {}  # simple Zobrist-like TT keyed by board hash
        self.KILLER_MOVES = {}         # optional killer move heuristic
        self.PV = []                   # principal variation (best line)
        # Piece values according to assignment points
        self.PV_VALUES = {
            'P': 20,   # pawn
            'B': 70,   # bishop
            'N': 70,   # knight
            'K': 10000 # king (very large)
        }
        self.CHECK_BONUS = 2

    def get_best_move(self):
        """
        Returns the best move for current board state.
        Uses iterative deepening with alpha-beta.
        """
        self.start_time = time.time()
        self.max_time = getattr(self.board, 'time_left', None)
        if self.max_time is None:
            self.max_time = 1.5

        best_move = None
        best_score = -float('inf')
        max_depth = self.depth
        try:
            for d in range(1, max_depth + 1):
                self.nodes_expanded = 0
                score, move = self._alphabeta_root(d, -float('inf'), float('inf'))
                if move is not None:
                    best_move, best_score = move, score
                if time.time() - self.start_time > min(self.max_time * self.time_limit, 0.95 * self.max_time):
                    break
        except TimeoutError:
            pass
        return best_move

    def evaluate_board(self):
        """Heuristic evaluation of the current board state."""
        return self._evaluate_board(self.board)

    # ---------------------------------------------------------------------
    # Helpers (same as before)
    # ---------------------------------------------------------------------
    def _board_hash(self, board):
        if hasattr(board, 'get_hash'): return board.get_hash()
        if hasattr(board, 'zobrist_hash'): return board.zobrist_hash()
        if hasattr(board, '__repr__'): return repr(board)
        if hasattr(board, 'board'):
            try: return tuple(tuple(row) for row in board.board)
            except Exception: return str(board)
        return str(board)

    def _get_legal_moves(self, board):
        if hasattr(board, 'get_legal_moves'): return list(board.get_legal_moves())
        if hasattr(board, 'legal_moves'):
            try: return list(board.legal_moves)
            except Exception: return board.legal_moves
        if hasattr(board, 'generate_legal_moves'): return list(board.generate_legal_moves())
        if hasattr(board, 'moves'): return list(board.moves())
        if hasattr(board, 'get_actions'): return list(board.get_actions())
        raise RuntimeError("Adapt _get_legal_moves to your Board API")

    def _make_move(self, board, move):
        if hasattr(board, 'push'): return board.push(move)
        if hasattr(board, 'make_move'): return board.make_move(move)
        if hasattr(board, 'apply_move'): return board.apply_move(move)
        if hasattr(board, 'after_move'): return board.after_move(move)
        if hasattr(board, 'move'): return board.move(move)
        raise RuntimeError("Adapt _make_move to your Board API")

    def _undo_move(self, board, move):
        if hasattr(board, 'pop'): return board.pop()
        if hasattr(board, 'undo_move'): return board.undo_move(move)
        if hasattr(board, 'unmake_move'): return board.unmake_move(move)
        return None

    def _is_terminal(self, board):
        if hasattr(board, 'is_game_over'): return board.is_game_over()
        if hasattr(board, 'game_over'): return board.game_over()
        if hasattr(board, 'is_terminal'): return board.is_terminal()
        try: return len(self._get_legal_moves(board)) == 0
        except Exception: return False

    def _evaluate_board(self, board):
        score = 0
        our_side = getattr(self, 'side', None)
        if our_side is None:
            if hasattr(board, 'turn'):
                our_side = board.turn
            else:
                our_side = 'W'

        def material_from_piece_list(piece_list):
            s = 0
            for p in piece_list:
                pt = getattr(p, 'type', None) or getattr(p, 'kind', None) or str(p)
                pt = str(pt).upper()
                if pt.startswith('P'): s += self.PV_VALUES['P']
                elif pt.startswith('B'): s += self.PV_VALUES['B']
                elif pt.startswith('N') or pt.startswith('KNT'): s += self.PV_VALUES['N']
                elif pt.startswith('K') and not pt.startswith('KN'): s += self.PV_VALUES['K']
            return s

        white_material = 0
        black_material = 0
        if hasattr(board, 'white_pieces') and hasattr(board, 'black_pieces'):
            try:
                white_material = material_from_piece_list(board.white_pieces)
                black_material = material_from_piece_list(board.black_pieces)
            except Exception:
                white_material = black_material = 0
        elif hasattr(board, 'board'):
            try:
                for row in board.board:
                    for cell in row:
                        if cell is None or cell == '.': continue
                        cstr = str(cell)
                        if cstr.isalpha() and len(cstr) == 1:
                            if cstr.isupper():
                                if cstr == 'P': white_material += self.PV_VALUES['P']
                                elif cstr == 'B': white_material += self.PV_VALUES['B']
                                elif cstr == 'N': white_material += self.PV_VALUES['N']
                                elif cstr == 'K': white_material += self.PV_VALUES['K']
                            else:
                                if cstr.lower() == 'p': black_material += self.PV_VALUES['P']
                                elif cstr.lower() == 'b': black_material += self.PV_VALUES['B']
                                elif cstr.lower() == 'n': black_material += self.PV_VALUES['N']
                                elif cstr.lower() == 'k': black_material += self.PV_VALUES['K']
            except Exception:
                pass

        try: mobility = len(self._get_legal_moves(board))
        except Exception: mobility = 0

        opponent_in_check = False
        if hasattr(board, 'is_in_check'):
            try: opponent_in_check = board.is_in_check(not getattr(board, 'turn', True))
            except Exception: opponent_in_check = False

        white_score = white_material
        black_score = black_material
        white_score += mobility * 1
        if opponent_in_check:
            if getattr(board, 'turn', 'W') == 'W':
                white_score += self.CHECK_BONUS
            else:
                black_score += self.CHECK_BONUS

        raw = white_score - black_score
        if hasattr(board, 'turn'):
            turn = getattr(board, 'turn')
            if isinstance(turn, bool): return raw if turn else -raw
            elif isinstance(turn, str): return raw if turn.upper().startswith('W') else -raw
        return raw

    def _alphabeta_root(self, depth, alpha, beta):
        legal_moves = self._get_legal_moves(self.board)
        if not legal_moves: return self.evaluate_board(), None
        best_move = None
        best_score = -float('inf')
        ordered = sorted(legal_moves, key=self._move_sort_key, reverse=True)
        for move in ordered:
            if time.time() - self.start_time > min(self.max_time * self.time_limit, 0.95 * self.max_time):
                raise TimeoutError()
            undo_info = None
            try:
                if hasattr(self.board, 'push'):
                    self.board.push(move); undo_info = True
                else:
                    new_board = self._make_move(self.board, move)
                    old_board = self.board; self.board = new_board; undo_info = old_board
            except: continue
            score = -self._alphabeta(depth - 1, -beta, -alpha)
            if undo_info is True and hasattr(self.board, 'pop'): self.board.pop()
            elif undo_info is not None: self.board = undo_info
            if score > best_score:
                best_score, best_move = score, move
            if best_score > alpha: alpha = best_score
            if alpha >= beta: break
        return best_score, best_move

    def _alphabeta(self, depth, alpha, beta):
        if time.time() - self.start_time > min(self.max_time * self.time_limit, 0.95 * self.max_time):
            raise TimeoutError()
        bh = self._board_hash(self.board)
        tt = self.transposition_table.get(bh)
        if tt is not None and tt.get('depth', -1) >= depth: return tt['value']
        if depth == 0 or self._is_terminal(self.board):
            val = self._evaluate_board(self.board)
            self.transposition_table[bh] = {'value': val, 'depth': depth}
            return val
        self.nodes_expanded += 1
        legal_moves = self._get_legal_moves(self.board)
        if not legal_moves:
            val = self._evaluate_board(self.board)
            self.transposition_table[bh] = {'value': val, 'depth': depth}
            return val
        ordered = sorted(legal_moves, key=self._move_sort_key, reverse=True)
        value = -float('inf')
        for move in ordered:
            if hasattr(self.board, 'push'):
                self.board.push(move); undo_info = True
            else:
                new_board = self._make_move(self.board, move)
                old_board = self.board; self.board = new_board; undo_info = old_board
            score = -self._alphabeta(depth - 1, -beta, -alpha)
            if undo_info is True and hasattr(self.board, 'pop'): self.board.pop()
            elif undo_info is not None: self.board = undo_info
            if score > value: value = score
            if value > alpha: alpha = value
            if alpha >= beta: break
        self.transposition_table[bh] = {'value': value, 'depth': depth}
        return value

    def _move_sort_key(self, move):
        score = 0
        for klist in self.KILLER_MOVES.values():
            if move in klist: score += 1000
        if hasattr(move, 'is_capture') and getattr(move, 'is_capture'):
            captured = getattr(move, 'captured', None)
            if captured:
                c = str(captured).upper()
                if c.startswith('P'): score += 20
                elif c.startswith('B'): score += 70
                elif c.startswith('N'): score += 70
                elif c.startswith('K'): score += 600
            else: score += 50
        elif hasattr(move, 'captured') and getattr(move, 'captured') is not None:
            c = str(getattr(move, 'captured')).upper()
            if c.startswith('P'): score += 20
            elif c.startswith('B'): score += 70
            elif c.startswith('N'): score += 70
            elif c.startswith('K'): score += 600
            else: score += 40
        else:
            try:
                s = str(move)
                if 'x' in s or 'X' in s: score += 40
            except: pass
        return score

