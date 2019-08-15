function run_demo(chunks, endings)
%
% Runs one of the demos defined in ../demo.
%

% Margins.
marg.comments = '   ';
marg.commands = '>> ';
marg.question = '   ';
marg.optional = '   Answer: ';

% Predefined answers with special meanings.
quit = {'quit', 'no'};
start = {'start', 'again'};
help = {'help'}
special = {quit{:} start{:} help{:}};

% Default questions and options.
defq = 'Continue?';
defo = {'yes', 'no'};
endq = 'Quit or start again?';
endo = {'quit', 'again'};

% Predefined speil for help and which is always added to chunk(1).comments.
speil = { ...
  'Type answers to the questions to move through the demo.', ...
  'Just hit the return key to get the default answer', ...
  'which is always first in the list of suggestions.', ...
  '', ...
  'The following additional commands are also valid, even', ...
  'though they may not be amongst the suggested answers:', ...
  '  ''quit'' allows you to quit at any time,', ...
  '  ''start'' returns you to the start of the demo,', ...
  '  ''help'' prints out this help message.', ...
  '', ...
  'You only need to type enough letters to disambiguate', ...
  'the answer. For example, ''q'' instead of ''quit'',', ...
  'or ''y'' instead of ''yes''.', ...
};

% Init.
answer = '';
i = 1;
fprintf('\n')

% Loop through chunks.
while i <= length(chunks)

  % Clear the screen.
  clc
  fprintf('\n')
  drawnow

  % Get a chunk.
  chunk = chunks(i);

  % Print the comments.
  comments = chunk.comments;
  if ~isempty(comments)
    if ischar(comments)
      comments = substitute(comments, answer);
      fprintf('%s%s\n', marg.comments, comments)
    else
      for j = 1:length(comments)
        comment = substitute(comments{j}, answer);
        fprintf('%s%s\n', marg.comments, comment)
      end
    end
    fprintf('\n')
  end

  % If this is chunk(1) print the speil as well.
  if i == 1
    for j = 1:length(speil)
      comment = speil{j};
      fprintf('%s%s\n', marg.comments, comment)
    end
    fprintf('\n')
  end

  % Print (unless silent [starts with '%']) and run the commands.
  commands = chunk.commands;
  if ~isempty(commands)
    nonshys = 0;
    if ischar(commands)
      commands = substitute(commands, answer);
      if strncmp(commands,'%',1)
        % Run silently.
        eval(commands(2:end))
      else
        % Don't be shy.
        fprintf('%s%s\n', marg.commands, commands)
        eval(commands)
        nonshys = nonshys + 1;
      end
    else
      for j = 1:length(commands)
        command = substitute(commands{j}, answer);
        if strncmp(command,'%',1)
          % Run silently.
          eval(command(2:end))
        else
          % Don't be shy.
          fprintf('%s%s\n', marg.commands, command)
          eval(command)
          nonshys = nonshys + 1;
        end
      end
    end
    if nonshys > 0
      fprintf('\n')
    end
  end

  % Supply a default question.
  question = chunk.question;
  optional = chunk.optional;
  if isempty(question)
    if i == length(chunks)
      question = endq;
      optional = endo;
    else
      question = defq;
      optional = defo;
    end
  end

  % Ask the question.
  if ischar(question)
    fprintf('%s%s\n', marg.question, question)
  else
    for j = 1:length(question)
      query = question{j};
      fprintf('%s%s\n', marg.question, query)
    end
  end

  % Get the answer.
  prompt = marg.optional;
  for j = 1:length(optional)
    option = optional{j};
    if j == length(optional)
      prompt = [prompt ' or '];
    elseif j > 1
      prompt = [prompt ', '];
    end
    prompt = [prompt option];
  end
  prompt = [prompt '? '];
  possibles = uniquely_combine(optional, special);
  while 1
    fprintf('\n')
    answer = lower(input(prompt, 's'));
    if isempty(answer)
      break;
    end
    k = find(strncmp(possibles, answer, length(answer)));
    if length(k) == 1
      answer = possibles{k};
      if isempty(find(strcmp(answer, help)))
        break;
      else
        % Print help message.
        fprintf('\n')
        for j = 1:length(speil)
          fprintf('%s%s\n', marg.comments, speil{j})
        end
        fprintf('%s%s\n', marg.comments, '')
        fprintf('%s%s\n', marg.comments, 'Continue?')
      end
    else
      fprintf('\n')
      if length(k) == 0
        fprintf('%sUnrecognised: try:', marg.comments)
        for j = 1:length(possibles)
          if j > 1
            if j == length(possibles)
              fprintf(' or')
            else
              fprintf(',')
            end
          end
          fprintf(' %s', possibles{j})
        end
        fprintf('\n')
      else
        fprintf('%sAmbiguous: could be:', marg.comments)
        ambigles = possibles(k);
        for j = 1:length(ambigles)
          if j > 1
            if j == length(ambigles)
              fprintf(' or')
            else
              fprintf(',')
            end
          end
          fprintf(' %s', ambigles{j})
        end
        fprintf('\n')
      end
    end
  end
  fprintf('\n')

  % Default answer.
  if isempty(answer)
    answer = optional{1};
  end

  % Deal with the answer.
  switch answer
  case quit
    break
  case start
    i = 1;
    answer = '';
  otherwise
    i = i + 1;
  end

end

% Clean up by executing the given command(s).
clc
if ~isempty(endings)
  if ischar(endings)
    eval(endings)
  else
    for j = 1:length(endings)
      eval(endings{j})
    end
  end
end

function str = substitute(str, answer)
% Substitue any occurrences of 'answer' in str with answer
% but only if answer is not empty.
if ~isempty(answer)
  found = findstr('answer', str);
  for i = length(found):-1:1
    k = found(i);
    str = [str(1:(k-1)) answer str((k+6):end)];
  end
end

function c = uniquely_combine(c1, c2)
% Combine cell arrays c1 and c2 into c but don't
% repeat elements in c2 which are already in c1.
c = c1;
for i = 1:length(c2)
  k = find(strcmp(c1, c2{i}));
  if isempty(k)
    c = {c{:} c2{i}};
  end
end
