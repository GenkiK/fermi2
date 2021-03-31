#include <bits/stdc++.h>

#define SORT(v, n) sort(v, v + n);
#define VSORT(v) sort(v.begin(), v.end());
#define size_t unsigned long long
#define ll long long
#define rep(i, a) for (int i = 0; i < (int)(a); i++)
#define repr(i, a) for (int i = (int)(a)-1; i >= 0; i--)
#define FOR(i, a, b) for (int i = (a); i < (b); i++)
#define FORR(i, a, b) for (int i = (int)(b)-1; i >= a; i--)
#define ALL(a) a.begin(), a.end()
using namespace std;

typedef pair<int, int> P;

class State
{
public:
  vector<int> v;
  int score;
  int num;

  State(int n)
  {
    v = vector<int>(n);
    score = 0;
    num = n;
    rep(i, n)
    {
      v[i] = i;
      score += i;
    }
  }

  State(vector<int> &input)
  {
    v = input;
    num = v.size();
    score = 0;
    rep(i, v.size())
    {
      score += v[i];
    }
  }

  void filePrint()
  {
    ofstream outputfile("state_list.txt");
    outputfile << score << " : [";
    rep(i, size())
    {
      outputfile << v[i] << ", ";
    }
    outputfile << "\b\b]" << endl;
    outputfile.close();
  }

  void print()
  {
    cout << score << " : [";
    rep(i, size())
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
  rep(i, b.v.size())
  {
    if (b.v[i] != a.v[i])
      return false;
  }
  return true;
}

bool operator<(const State &a, const State &b)
{
  if (a.score == b.score)
  {
    rep(i, b.v.size())
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
  set<State> s;
  int lim_score;
  int n;

public:
  Fermi(int n, int lim_size = 10)
  {
    init(n, lim_size);
  }

  Fermi()
  {
    init(3);
  }

  void printAll()
  {
    cout << "ε : state" << endl;
    for (auto state : s)
    {
      state.filePrint();
      // state.print();
    }
  }

  void fileCountPrint()
  {
    map<int, int> m;
    for (auto state : s)
    {
      m[state.score]++;
    }
    ofstream outputfile("state_count" + to_string(n) + ".csv");
    outputfile << "level,nums" << endl;
    for (auto ma : m)
    {
      outputfile << ma.first << ", " << ma.second << endl;
    }
    outputfile.close();
  }

private:
  void init(int n, int lim_size = 10)
  {
    this->n = n;
    State state(n);
    lim_score = state.score + lim_size;
    init(state.v);
  }

  void init(vector<int> v)
  {
    State state(v);
    if (!array_is_unique(v))
      return;
    if (state.score > lim_score)
      return;
    if (s.emplace(state).second)
    {
      rep(i, state.size())
      {
        state.increment(i);
        init(state.v);
        state.decrement(i);
      }
    };
  }

  bool array_is_unique(const vector<int> &v)
  {
    rep(i, v.size() - 1)
    {
      if (v[i] == v[i + 1])
        return false;
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
    cout << "       n = ";
    cin >> n;
    cout << "lim_size = ";
    cin >> lim_size;
  }
  Fermi f(n, lim_size);
  // f.printAll();
  f.fileCountPrint();
}
