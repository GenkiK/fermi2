#include <bits/stdc++.h>

#define size_t unsigned long long
#define ll long long
using namespace std;

typedef pair<int, int> P;

class State
{
public:
  vector<int> v; // それぞれの粒子のエネルギー
  int score;     // 総エネルギー数
  int num;       // 電子の数

  State(int n)
  {
    v = vector<int>(n);
    score = 0;
    num = n;
    for (int i = 0; i < n; i++)
    {
      v[i] = i;
      score += i;
    }
  }

  State(vector<int> input)
  {
    num = input.size();
    score = 0;
    v = input;
    for (int i = 0; i < num; i++)
    {
      score += input[i];
    }
  }

  // Refactor as exportStateList() in Fermi class.
  // void filePrint()
  // {
  //   ofstream outputfile("output/state_list.txt");
  //   outputfile << score << "ε: [";
  //   for (int i = 0; i < size(); i++)
  //   {
  //     outputfile << v[i] << ", ";
  //   }
  //   outputfile << "]\n" << endl;
  //   outputfile.close();
  // }

  void print()
  {
    cout << score << "ε: [";
    for (int i = 0; i < size(); i++)
    {
      cout << v[i] << ", ";
    }
    cout << "\b\b]" << endl;
  }

  bool increment(int idx)
  {
    if (this->size() <= idx)
    {
      cout << "State: idx(" << idx << ") is bigger than size(" << size() << ")" << endl;
      return false;
    }
    this->v[idx]++;
    return true;
  }

  bool decrement(int idx)
  {
    if (this->size() <= idx)
    {
      cout << "State: idx is bigger than size()" << endl;
      return false;
    }
    this->v[idx]--;
    return true;
  }

  int size()
  {
    return num;
  }
};

bool operator==(const State &a, const State &b)
{
  if (b.v.size() != a.v.size())
    return false;
  for (int i = 0; i < b.v.size(); i++)
  {
    if (b.v[i] != a.v[i])
      return false;
  }
  return true;
}

bool operator!=(const State &a, const State &b)
{
  return !(a == b);
}

bool operator<(const State &a, const State &b)
{
  if (a.score == b.score)
  {
    for (int i = 0; i < b.v.size(); i++)
    {
      if (a.v[i] != b.v[i])
        return (a.v[i] < b.v[i]);
    }
  }
  return a.score < b.score;
}

bool operator>(const State &a, const State &b) { return b < a; }
bool operator<=(const State &t1, const State &t2) { return !(t1 > t2); }
bool operator>=(const State &t1, const State &t2) { return !(t1 < t2); }

class Fermi
{
  set<State> s;  // 各ステートを管理
  int lim_score; // 総エネルギーの最大値
  int n;         // 電子の数

public:
  Fermi(int n, int lim_size = 10)
  {
    //O()
    init(n, lim_size);
  }

  Fermi()
  {
    init(3);
  }

  void printStates()  // 旧printAll()
  {
    cout << "ε : state" << endl;
    for (auto state : s)
    {
      // state.filePrint();
      state.print();
    }
  }

  void exportStatesTxt()
  /**
   * Stateの一覧を、"score ε: [v[0], v[1],...]" という形で、txt形式として出力する関数
  */
  {
    cout << "ε : state" << endl;
    ofstream outputfile("output/state_list" + to_string(n) + ".txt");
    for (auto state : s)
    {
      outputfile << state.score << "ε: [";
      for (int i = 0; i < state.size(); i++)
      {
        outputfile << state.v[i];
        if (i == state.size()-1) outputfile << "]" << endl;
        else outputfile << ", ";
      }
    }
    outputfile.close();
  }

  void exportStatesCsv()
  /**
   * Stateの一覧を、"score, v[0], v[1], ..., v[n-1]" という形で、cSV形式として出力する関数
   */
  {
    cout << "score, v[0], v[1], ..., v[n-1]" << endl;
    ofstream outputfile("output/states" + to_string(n) + ".csv");
    // header
    outputfile << "score";
    for (int i = 0; i < n; i++)
    {
      outputfile << ", v[" << to_string(i) << "]";
    }
    outputfile << endl;

    for (auto state : s)
    {
      outputfile << state.score << ", ";
      for (int i = 0; i < state.size(); i++)
      {
        outputfile << state.v[i];
        if (i == state.size()-1) outputfile << endl;
        else outputfile << ", ";
      }
    }
    outputfile.close();
  }

  void exportStateCount()  // 旧fileCountPrint()
  /**
   * 同じ総エネルギー数(score)をもつStateの数を、csv形式で出力する関数
  */
  {
    map<int, int> m;
    for (auto state : s)
    {
      m[state.score]++;
    }
    ofstream outputfile("output/state_count" + to_string(n) + ".csv");
    outputfile << "score, nums" << endl;
    for (auto ma : m)
    {
      outputfile << ma.first << ", " << ma.second << endl;
    }
    outputfile.close();
  }

  void countPair() // O(準位数 * N * lim_size)
  {
    vector<vector<int>> pair(lim_score + 1, vector<int>(lim_score + 1, 0));
    vector<vector<int>> diff(n - 1, vector<int>(n - 1));
    for (auto state : s)
    {
      cout << "state.v: [";
      for (auto state_i : state.v) cout << state_i << ", ";
      cout << "]" << endl;
      for (int i = 0; i < n - 1; i++)
      {
        for (int j = i + 1; j < n; j++)
        {
          diff[i][j] = state.v[j] - state.v[i];  // 各Stateごとにdiffは初期化される
        }
      }

      // いま考えているStateのスコア(state.score)から別のスコア(state.score+k)をもつStateに遷移するパターン数は電子をn個のうちいずれかを遷移させるので、遷移可能なのは最大nパターン。(ここではstate.score + k よりも大きい配置に遷移することも含めて考えてる)
      for (int k = 1; k + state.score <= lim_score; k++)
      {
        pair[state.score][state.score + k] += n;
      }

      for (int i = 0; i < n - 1; i++)
      {
        for (int j = i + 1; j < n; j++)
        {
          if (state.score + diff[i][j] <= lim_score)
            pair[state.score][state.score + diff[i][j]]--;
        }
      }
    }

    ofstream outputfile("output/pair_count" + to_string(n) + ".csv");
    for (int i = 0; i < pair.size(); i++)
    {
      // ヘッダー
      if (i == 0)
      {
        for (int j = 0; j < pair[i].size(); j++)
        {
          outputfile << j;
          if (j == pair[i].size()-1) outputfile << endl;
          else outputfile << ",";
        }
      }
      for (int j = 0; j < pair[i].size(); j++)
      {
        outputfile << pair[i][j];
        if (j == pair[i].size()-1) outputfile << endl;
        else outputfile << ",";
      }
    }
    outputfile.close();
    return;
  }

  void secondPair()
  {
    cout << "secondPair" << endl;
    int lim = lim_score + 1;
    int num = 0;
    int pairs[lim][lim][8];
    vector<string> strs{"+++", "++-", "+-+", "+--", "-++", "-+-", "--+", "---"};

    // TODO: outputディレクトリを存在しなければ作るような関数を実装する。
    ofstream outputfile("output/pair3D2_count" + to_string(n) + ".csv");
    for (auto str : strs)
      outputfile << str << ",";
    outputfile << "\n";
    for (auto ite1 = s.begin(); ite1 != s.end(); ++ite1)
    {
      if (num != ite1->score)
      {
        for (int i = 0; i < lim; i++)
        {
          for (int j = 0; j < lim; j++)
          {
            for (int l = 0; l < 8; l++)
            {
              pairs[i][j][l] = 0;
            }
          }
          num = ite1->score;
          cout << num << " / " << lim << endl;
        }
        for (auto ite2 = next(s.end(), -1); ite2 != ite1; --ite2)
        {
          if (ite1->score == ite2->score)
            break;
          unordered_map<int, int> temp;
          for (int i = 0; i < n; i++)
          {
            temp[ite1->v[i]]++;
            temp[ite2->v[i]]++;
          }
          for (auto ite3 = next(s.end(), -1); ite3 != ite2; --ite3)
          {
            if (ite2->score == ite3->score)
              break;
            int idx = (int)temp.size() != n + 1 ? 4 : 0;
            unordered_map<int, int> temp1, temp2;
            for (int i = 0; i < n; i++)
            {
              temp1[ite3->v[i]]++;
              temp1[ite2->v[i]]++;
              temp2[ite1->v[i]]++;
              temp2[ite3->v[i]]++;
            }
            idx += (int)temp1.size() != n + 1 ? 2 : 0;
            idx += (int)temp2.size() != n + 1 ? 1 : 0;
            pairs[ite2->score][ite3->score][idx]++;
          }
        }
        if (next(ite1, 1) == s.end() || ite1->score != next(ite1, 1)->score)
        {
          for (int j = ite1->score + 1; j < lim; j++)
          {
            for (int k = j + 1; k < lim; k++)
            {
              outputfile << ite1->score << "," << j << "," << k << ",";
              for (int l = 0; l < 8; l++)
              {
                outputfile << pairs[j][k][l] << ",";
              }
              outputfile << "\n";
            }
          }
        }
      }

      outputfile.close();
    }
  }

  void makePair()
  {
    vector<vector<int>> pairVec(lim_score + 1, vector<int>(lim_score + 1, 0));
    int num = 0;
    for (auto a : s)
    {
      if (num != a.score)
      {
        cout << num << " / " << lim_score << endl;
        num = a.score;
      }
      for (auto b : s)
      {
        if (a.score <= b.score)
        {
          continue;
        }
        if (is_connected(a, b))
        {
          pairVec[a.score][b.score]++;
        }
      }
    }
    ofstream outputfile("output/pairVec_count" + to_string(n) + ".csv");
    for (int i = 0; i < pairVec.size(); i++)
    {
      // ヘッダー
      if (i == 0) {
        for (int j = 0; j < pairVec[i].size(); j++)
        {
          outputfile << j;
          if (j == pairVec[i].size() - 1)
            outputfile << endl;
          else
            outputfile << ",";
        }
      }
      for (int j = 0; j < pairVec[i].size(); j++)
      {
        outputfile << pairVec[i][j];
        if (j == pairVec[i].size() - 1)
          outputfile << endl;
        else
          outputfile << ",";
      }
    }
    outputfile.close();
    return;
  }

private:
  void init(int n, int lim_size = 10)
  {
    this->n = n;
    State state(n);
    lim_score = state.score + lim_size;
    init(state.v);
  }

  /// 深さ優先探索でステートを列挙
  void init(const vector<int> &v)
  {
    State state(v);
    if (!array_is_unique(v))
      return;
    if (state.score > lim_score)
      return;
    if (s.emplace(state).second)
    {
      for (int i = 0; i < state.size(); i++)
      {
        state.increment(i);
        init(state.v);
        state.decrement(i);
      }
    };
  }

  bool array_is_unique(const vector<int> &v)
  {
    for (int i = 0; i < v.size() - 1; i++)
    {
      if (v[i] == v[i + 1])
        return false;
    }

    return true;
  }

  bool is_connected(const State &a, const State &b)
  {
    /*
    024
    034
    */
    int i = 0, j = 0;
    int diff = 0;
    while (i < a.num && j < b.num)
    {
      if (a.v[i] == b.v[i])
      {
        i++;
        j++;
      }
      else if (a.v[i] < b.v[j])
      {
        if (++diff > 1)
          return false;
        i++;
        if (a.v[i] > b.v[i])
          j++;
      }
      else
      {
        if (++diff > 1)
          return false;
        j++;
        if (a.v[i] > b.v[i])
          j++;
      }
    }
    return true;
  }
};

int main(int argc, char **argv)
{
  int n, lim_size;
  if (argc > 1)
  {
    n = atoi(argv[1]);
    lim_size = atoi(argv[2]);
  }
  else
  {
    cout << "粒子数nを入力" << endl;
    cin >> n;
    cout << "基底準位からのエネルギーの上がり幅lim_sizeを入力" << endl;
    cin >> lim_size;
  }
  Fermi f(n, lim_size);
  cout << endl;
  // f.secondPair();
  // f.exportStateList();
  // f.countPair();
  // f.makePair();
  f.exportStatesCsv();
}
